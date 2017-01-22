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
import logging
import random
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D

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
                 frames_per_action=1,
                 observe_state_length=50000,
                 observations_per_state=4):

        # Constants
        self._actions = possible_actions
        self._checkpointpath = checkpointpath
        self._summarypath = summarypath
        self._gamma = gamma
        self._epsilon = epsilon_init
        self._epsilon_final = epsilon_final
        self._epsilon_anneal_over_frames = (epsilon_init - epsilon_final) / float(epsilon_anneal_over_frames)
        self._batch_size = batch_size
        self._frames_per_action = frames_per_action
        self._q_target_update_interval = q_target_update_interval
        self._observe_state_length = observe_state_length
        self._observations_per_state = observations_per_state

        # Counters
        self._timestep = 0
        self._prev_timestep = 0
        self._prev_timestamp = time.clock()

        self._replay_memory = RingBuffer(replay_memory_size, dtype=np.ndarray)
        self._current_state = None
        self._is_initialized = False

    def _init_for_first_observation(self, observation):
        self._is_initialized = True
        self._current_state = np.stack((observation,) * self._observations_per_state, axis=-1)
        self._init_convnets(observation)

    def _init_convnets(self, observation):
        # pylint: disable=attribute-defined-outside-init
        input_dimension = list(observation.shape[:2]) + [self._observations_per_state]

        self.q_model = Sequential()
        self.q_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',
                                       input_shape=input_dimension, dim_ordering='tf'))
        self.q_model.add(Activation('relu'))

        self.q_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.q_model.add(Activation('relu'))
        self.q_model.add(Dropout(0.25))

        self.q_model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
        self.q_model.add(Activation('relu'))
        self.q_model.add(Dropout(0.25))

        self.q_model.add(Flatten())
        self.q_model.add(Dense(512))
        self.q_model.add(Activation('relu'))
        self.q_model.add(Dropout(0.5))

        self.q_model.add(Dense(self._actions))
        self.q_model.add(Activation('linear'))

        self.q_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        # Copy q_model into q_model_t
        self.q_model_t = copy.copy(self.q_model)
        self._copy_target_qnetwork()

    def get_action(self, next_observation, action, reward, terminal):
        '''
        TODO: write a docstring
        '''
        assert len(next_observation.shape) == 3 and \
               next_observation.shape[2] == 1, 'Only single channel frames supported!'

        next_observation.shape = next_observation.shape[:2]

        if not self._is_initialized:
            self._init_for_first_observation(next_observation)

        self._set_perception(next_observation, action, reward, terminal)
        return self._get_action()

    @property
    def dqn_state(self):
        '''
        TODO: Write a docstring
        '''
        if self._timestep <= self._observe_state_length:
            return "observe"
        elif self._timestep > self._observe_state_length and \
             self._timestep <= self._observe_state_length + self._epsilon_anneal_over_frames:
            return "explore"
        else:
            return 'train'

    def _copy_target_qnetwork(self):
        self.q_model_t.set_weights(self.q_model.get_weights())

    def _set_perception(self, next_observation, action, reward, terminal):
        # Copy current-state (excluding oldest frame)
        new_state = np.append(self._current_state[:, :, 1:], np.expand_dims(next_observation, axis=-1), axis=-1)
        self._replay_memory.append((self._current_state, action, reward, new_state, terminal))

        if self.dqn_state != 'observe':
            self._train_qnetwork()

        self._current_state = new_state
        self._timestep += 1

    def _get_action(self):
        qvalue = self.q_model.predict_on_batch(np.asarray([self._current_state]))[0]

        # Assume action=0 means do nothing
        action = 0
        if self._timestep % self._frames_per_action == 0:
            if random.random() <= self._epsilon:
                action = random.randrange(self._actions)
            else:
                action = np.argmax(qvalue)

        if self.dqn_state != 'observe' and self._epsilon > self._epsilon_final:
            self._epsilon -= self._epsilon_anneal_over_frames

        return action


    def _train_qnetwork(self):
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
        qvalue_t_batch = self.q_model_t.predict_on_batch(np.asarray(next_state_batch))

        for terminal, reward, action, qvalue_t in zip(terminal_batch, reward_batch, action_batch, qvalue_t_batch):
            if not terminal:
                reward = reward + self._gamma * np.max(qvalue_t)
            qvalue_t[action] = reward

        self.q_model.fit(np.asarray(state_batch), qvalue_t_batch, verbose=0, batch_size=self._batch_size)

        if self._timestep % self._q_target_update_interval == 0:
            self._copy_target_qnetwork()
