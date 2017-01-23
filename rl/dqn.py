#!/usr/bin/env python
'''
Copyright (c) 2017, Jesper Derehag <jderehag@hotmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import copy
import datetime
import logging
import os
import random
import time

from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D

import tensorflow as tf
import numpy as np

from util.ringbuffer import RingBuffer

logger = logging.getLogger(__name__)


class DQN(object):
    '''
    TODO: write a docstring
    '''
    def __init__(self, possible_actions, checkpointpath, summarypath,
                 gamma=0.95,
                 replay_memory_size=1000000,
                 epsilon_init=1.0,
                 epsilon_final=0.1,
                 epsilon_anneal_over_frames=1000000.0,
                 batch_size=32,
                 q_target_update_interval=10000,
                 frames_per_action=4,
                 observe_state_length=50000,
                 observations_per_state=4):

        #############################################
        # Constants
        #############################################
        self._actions = possible_actions
        self._checkpointpath = checkpointpath
        self._summarypath = os.path.join(summarypath, str(datetime.datetime.now()))
        self._gamma = gamma
        self._epsilon = epsilon_init
        self._epsilon_final = epsilon_final
        self._epsilon_anneal_over_frames = (epsilon_init - epsilon_final) / float(epsilon_anneal_over_frames)
        self._batch_size = batch_size
        self._frames_per_action = frames_per_action
        self._q_target_update_interval = q_target_update_interval
        self._observe_state_length = observe_state_length
        self._observations_per_state = observations_per_state

        self._replay_memory = RingBuffer(replay_memory_size, dtype=np.ndarray)

        #############################################
        # Counters
        #############################################
        self._timestep = 0
        self._prev_timestep = 0
        self._prev_timestamp = time.clock()
        self._episode_score = 0
        self._current_score = 0

        tf.scalar_summary(r'game/score',
                          tf.Variable(initial_value=0, dtype=tf.int64, name='score'), name='score')
        tf.scalar_summary(r'game/fps',
                          tf.Variable(initial_value=0.0, dtype=tf.float32, name='fps'), name='fps')
        tf.scalar_summary(r'agent/epsilon (exploration probability)',
                          tf.Variable(initial_value=0.0, dtype=tf.float32, name='epsilon'), name='epsilon')
        self._summaries = tf.merge_all_summaries()
        self._summarywriter = tf.train.SummaryWriter(self._summarypath)


        ####################################################################
        # Models
        # Models are created at first observation (lazy initialization)
        ####################################################################
        self._is_initialized = False
        self._current_state = None
        self._q_model = None
        self._q_model_t = None

        ####################################################################
        # Summaries and checkpoints
        ####################################################################
        self._tensorboard = TensorBoard(log_dir=self._summarypath, write_graph=False)
        try:
            os.makedirs(self._checkpointpath)
            os.makedirs(self._summarypath)
        except OSError:
            pass

    def get_action(self, observation, action, reward, terminal):
        '''
        Gets an action given an observation, action, reward, terminal

        Essentially done in 4 steps:
            1. Store all observations in a fifo queue (experience replay)
            2. Trains a model to predict the reward using data from the fifo queue
            3. Predict the expected reward for all possible actions given this observation
            4. Do epsilon greedy selection of action (select random action given some probability epsilon)

        Args:
            observation (np.ndarray, shape(height,width,z=1)): The current observation (frame)
            action (int): The action performed prior to the current observation
            reward(float): The reward given this action and observation
            terminal(Boolean): End of episode, i,e if the game terminated

        Returns:
            action(int): The selected action as an index into the list of possible actions
        '''
        self._timestep += 1

        self._update_counters(reward, terminal)

        if self._timestep % self._frames_per_action != 0:
            return action

        observation = self._remove_channels(observation)

        if not self._is_initialized:
            self._init_for_first_observation(observation)

        self._new_state(observation, action, reward, terminal)

        if self.state != 'observe':
            self._train()

        action = self._epsilon_greedy_action()

        return action

    @property
    def state(self):
        '''
        The overall state of the agent

        The agent works in 3 phases:
            1. observe (only populate replay memory)
            2. explore (epilon-greedy annealation)
            3. train (no more epsilon annealation, play and explore with _epsilon_final probability)

        Returns:
            state(str): observe|explore|train
        '''
        if self._timestep <= self._observe_state_length:
            return "observe"
        elif self._timestep > self._observe_state_length and \
             self._timestep <= self._observe_state_length + self._epsilon_anneal_over_frames:
            return "explore"
        else:
            return 'train'

    def _train(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        # Would have liked to use np.transpose here but since
        # replay_memory.dtype=object, this is not doable for now
        for sample in self._replay_memory.sample(self._batch_size):
            state_batch.append(sample[0])
            action_batch.append(sample[1])
            reward_batch.append(sample[2])
            next_state_batch.append(sample[3])
            terminal_batch.append(sample[4])

        # Calculate discounted future reward, and update with new action-reward
        qvalue_t_batch = self._q_model_t.predict_on_batch(np.asarray(next_state_batch))

        for terminal, reward, action, qvalue_t in zip(terminal_batch, reward_batch, action_batch, qvalue_t_batch):
            if not terminal:
                reward = reward + self._gamma * np.max(qvalue_t)
            qvalue_t[action] = reward

        self._q_model.fit(np.asarray(state_batch), qvalue_t_batch, verbose=0,
                          batch_size=self._batch_size, callbacks=[self._tensorboard])

        if self._timestep % self._q_target_update_interval == 0:
            self._copy_target_qnetwork()
            self._q_model.save(os.path.join(self._checkpointpath, str(datetime.datetime.now()) + '.h5'))

    def _new_state(self, observation, action, reward, terminal):
        # Copy current-state (excluding oldest frame) and append new observation at end
        # Future: optimize by using np.roll() and no copy!
        new_state = np.append(self._current_state[:, :, 1:], np.expand_dims(observation, axis=-1), axis=-1)
        self._replay_memory.append((self._current_state, action, reward, new_state, terminal))
        self._current_state = new_state

    def _epsilon_greedy_action(self):
        if random.random() <= self._epsilon:
            action = random.randrange(self._actions)
        else:
            qvalue = self._q_model.predict_on_batch(np.asarray([self._current_state]))[0]
            action = np.argmax(qvalue)

        # Update epsilon
        if self.state == 'explore':
            self._epsilon -= self._epsilon_anneal_over_frames

        return action

    def _init_for_first_observation(self, observation):
        self._is_initialized = True
        self._current_state = np.stack((observation,) * self._observations_per_state, axis=-1)
        self._init_convnets(observation)

    def _init_convnets(self, observation):
        checkpoints = sorted(os.listdir(self._checkpointpath))
        if len(checkpoints) > 0:
            checkpoint = os.path.join(self._checkpointpath, checkpoints[-1])
            logger.warn('Loading checkpoint:%s', checkpoint)
            self._q_model = load_model(checkpoint)
        else:
            input_dimension = list(observation.shape[:2]) + [self._observations_per_state]

            self._q_model = Sequential()
            self._q_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                                            input_shape=input_dimension, dim_ordering='tf'))
            self._q_model.add(Activation('relu'))

            self._q_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
            self._q_model.add(Activation('relu'))

            self._q_model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
            self._q_model.add(Activation('relu'))

            self._q_model.add(Flatten())
            self._q_model.add(Dense(512))
            self._q_model.add(Activation('relu'))

            self._q_model.add(Dense(self._actions))
            self._q_model.add(Activation('linear'))

            self._q_model.compile(loss='mean_squared_error', optimizer='rmsprop')

        # Copy _q_model into _q_model_t
        self._q_model_t = copy.copy(self._q_model)
        self._copy_target_qnetwork()

    def _copy_target_qnetwork(self):
        self._q_model_t.set_weights(self._q_model.get_weights())

    def _update_counters(self, reward, terminal):
        self._current_score += reward
        if terminal:
            self._episode_score = self._current_score
            self._current_score = 0

        if self._timestep % 100 == 0:
            new_timestamp = time.clock()
            fps = (self._timestep - self._prev_timestep) / (new_timestamp - self._prev_timestamp)
            self._prev_timestamp = new_timestamp
            self._prev_timestep = self._timestep

            with tf.Session() as sess:
                counters = {'fps:0': fps, 'epsilon:0': self._epsilon, 'score:0': self._episode_score}
                result = sess.run([self._summaries], feed_dict=counters)
                self._summarywriter.add_summary(result[0], global_step=self._timestep)

    @staticmethod
    def _remove_channels(observation):
        assert len(observation.shape) == 3 and observation.shape[2] == 1, 'Only single channel frames supported!'
        observation.shape = observation.shape[:2]
        return observation

