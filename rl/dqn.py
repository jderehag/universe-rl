#!/usr/bin/env python
'''
Copyright (c) 2015, Jesper Derehag <jderehag@hotmail.com>
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
import os
import time
import datetime
import random
import logging
from collections import deque

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)

class DQN(object):
    '''
    TODO: write a docstring
    '''
    def __init__(self, possible_actions,
                 checkpointpath=None,
                 summarypath=None,
                 gamma=0.95,
                 replay_memory_size=1000000,
                 epsilon_init=1.0,
                 epsilon_final=0.1,
                 epsilon_anneal_over_frames=1000000,
                 minibatch=32,
                 q_target_update_interval=10000,
                 frames_per_action=1,
                 observe_state_length=50000,
                 observations_per_state=4):

        # Constants
        self._actions = possible_actions
        self._gamma = gamma
        self._replay_memory_size = replay_memory_size
        self._epsilon_init = epsilon_init
        self._epsilon_final = epsilon_final
        self._epsilon_anneal_over_frames = epsilon_anneal_over_frames
        self._minibatch_size = minibatch
        self._frames_per_action = frames_per_action
        self._q_target_update_interval = q_target_update_interval
        self._observe_state_length = observe_state_length
        self._observations_per_state = observations_per_state

        # Counters
        self._timestep = 0
        self._prev_timestep = 0
        self._prev_timestamp = time.clock()

        self._fps = tf.Variable(initial_value=0.0, dtype=tf.float32, name='frames_per_second')
        self._game_score = tf.Variable(initial_value=0.0, dtype=tf.float32, name='game_score')
        self._game_score_tmp = 0.0
        self._loss_function = tf.Variable(initial_value=0.0, dtype=tf.float32, name='loss')
        self._epsilon = tf.Variable(initial_value=epsilon_init, dtype=tf.float32, name='epsilon')

        self._replay_memory = deque()
        self._current_state = None

        self._session = tf.InteractiveSession()
        self._saver = tf.train.Saver()

        self._init_summaries(summarypath)
        self._init_checkpoints(checkpointpath)
        self._session.run(tf.initialize_all_variables())

    def _init_checkpoints(self, checkpointpath):
        if checkpointpath is not None:
            try:
                os.makedirs(checkpointpath)
            except OSError:
                pass
            finally:
                self._checkpoint = tf.train.get_checkpoint_state(checkpointpath)
                if self._checkpoint and self._checkpoint.model_checkpoint_path:
                    self._saver.restore(self._session, self._checkpoint.model_checkpoint_path)
                    logging.warning("Loaded: %s", self._checkpoint.model_checkpoint_path)
                else:
                    logging.info("Could not find any snapshot, starting training from null state")

    def _init_summaries(self, summarypath):
        if summarypath is not None:
            tf.scalar_summary(r'overview/cost (loss function)', self._loss_function)
            tf.scalar_summary(r'overview/epsilon (exploration probability)', self._epsilon)
            tf.scalar_summary(r'overview/game_score', self._game_score)
            tf.scalar_summary(r'performance/frames_per_second', self._fps)
            self._summaries = tf.merge_all_summaries()
            self._summarywriter = tf.train.SummaryWriter(os.path.join(summarypath, str(datetime.datetime.now())),
                                                         self._session.graph)

    def _init_for_first_observation(self, observation):
        self._current_state = np.stack((observation,) * self._observations_per_state, axis=-1)
        self._init_convnets(observation)

    def _init_convnets(self, observation):
        # pylint: disable=attribute-defined-outside-init
        input_dimension = list(observation.shape) + [self._observations_per_state]

        # init Q network
        self._stateinput, \
        self._qvalue, \
        self._w_conv1, \
        self._b_conv_1, \
        self._w_conv_2, \
        self._b_conv2, \
        self._w_conv_3, \
        self._b_conv3, \
        self._w_fc_1, \
        self._b_fc1, \
        self._w_fc_2, \
        self._b_fc2 = self._create_q_network(input_dimension=input_dimension)

        # init Target Q Network
        self._stateinput_t, \
        self._qvalue_t, \
        self._w_conv_1_t, \
        self._b_conv_1_t, \
        self._w_conv_2_t, \
        self._b_conv_2_t, \
        self._w_conv_3_t, \
        self._b_conv_3_t, \
        self._w_fc_1_t, \
        self._b_fc_1_t, \
        self._w_fc_2_t, \
        self._b_fc_2_t = self._create_q_network(input_dimension=input_dimension)

        with tf.name_scope('copy_target_qnetwork'):
            self._copy_target_qnetwork_operation = [self._w_conv_1_t.assign(self._w_conv1),
                                                    self._b_conv_1_t.assign(self._b_conv_1),
                                                    self._w_conv_2_t.assign(self._w_conv_2),
                                                    self._b_conv_2_t.assign(self._b_conv2),
                                                    self._w_conv_3_t.assign(self._w_conv_3),
                                                    self._b_conv_3_t.assign(self._b_conv3),
                                                    self._w_fc_1_t.assign(self._w_fc_1),
                                                    self._b_fc_1_t.assign(self._b_fc1),
                                                    self._w_fc_2_t.assign(self._w_fc_2),
                                                    self._b_fc_2_t.assign(self._b_fc2)]
        with tf.name_scope('train_qnetwork'):
            self._action_input = tf.placeholder("float", [None, self._actions], name='action_input')
            self._y_input = tf.placeholder("float", [None], name='y_input')
            q_action = tf.reduce_sum(tf.mul(self._qvalue, self._action_input), reduction_indices=1, name='q_action')
            self._cost = tf.reduce_mean(tf.square(self._y_input - q_action), name='loss_function')
            self._train_step = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self._cost)

        self._session.run(tf.initialize_all_variables())

    def get_action(self, next_observation, action, reward, terminal):
        '''
        TODO: write a docstring
        '''
        if self._current_state is None:
            self._init_for_first_observation(next_observation)

        self._set_perception(next_observation, action, reward, terminal)
        return self._get_action()

    def get_dqn_state(self):
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

    def _set_perception(self, next_observation, action, reward, terminal):
        new_state = np.append(self._current_state[:, :, 1:], np.expand_dims(next_observation, axis=-1), axis=-1)
        self._replay_memory.append((self._current_state, action, reward, new_state, terminal))

        if len(self._replay_memory) > self._replay_memory_size:
            self._replay_memory.popleft()

        if self.get_dqn_state() != 'observe':
            self._train_qnetwork()

        if self._timestep % 100 == 0:
            self._write_counters()

        self._game_score_tmp += reward
        if terminal:
            self._game_score.assign(self._game_score_tmp).op.run()
            self._game_score_tmp = 0.0

        self._current_state = new_state
        self._timestep += 1

    def _get_action(self):
        qvalue = self._qvalue.eval(feed_dict={self._stateinput:[self._current_state]})[0]
        epsilon = self._epsilon.eval()

        # Assume action=0 means do nothing
        action = 0
        if self._timestep % self._frames_per_action == 0:
            if random.random() <= epsilon:
                action = random.randrange(self._actions)
            else:
                action = np.argmax(qvalue)

        # change episilon
        if epsilon > self._epsilon_final and self.get_dqn_state() != 'observe':
            self._epsilon.assign_sub((self._epsilon_init - self._epsilon_final) / \
                                      self._epsilon_anneal_over_frames).op.run()
        return action

    def _create_q_network(self, input_dimension):
        with tf.name_scope('input_layer'):
            stateinput = tf.placeholder("float", [None] + input_dimension, name='stateinput')

        # hidden layers
        with tf.name_scope('conv1'):
            w_conv1 = self._weight_variable([8, 8, 4, 32], name='conv1_weights')
            b_conv1 = self._bias_variable([32], name='conv1_bias')
            h_conv1 = tf.nn.relu(self._conv2d(stateinput, w_conv1, 4) + b_conv1, name='conv1_relu')
            # h_pool1 = self._max_pool_2x2(h_conv1)
        with tf.name_scope('conv2'):
            w_conv2 = self._weight_variable([4, 4, 32, 64], name='conv2_weights')
            b_conv2 = self._bias_variable([64], name='conv2_bias')
            h_conv2 = tf.nn.relu(self._conv2d(h_conv1, w_conv2, 2) + b_conv2, name='conv2_relu')

        with tf.name_scope('conv3'):
            w_conv3 = self._weight_variable([3, 3, 64, 64], name='conv3_weights')
            b_conv3 = self._bias_variable([64], name='conv3_bias')
            h_conv3 = tf.nn.relu(self._conv2d(h_conv2, w_conv3, 1) + b_conv3, name='conv3_relu')
            # h_conv3_shape = h_conv3.get_shape().as_list()
            # print "dimension:", h_conv3_shape[1] * h_conv3_shape[2] * h_conv3_shape[3]
            h_conv3_flat = tf.reshape(h_conv3, [-1, 3136], name='conv3_flatten')

        with tf.name_scope('fc1'):
            w_fc1 = self._weight_variable([3136, 512], name='fc1_weights')
            b_fc1 = self._bias_variable([512], name='fc1_bias')
            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, w_fc1) + b_fc1, name='fc1_relu')

        with tf.name_scope('fc2'):
            w_fc2 = self._weight_variable([512, self._actions], name='fc2_weights')
            b_fc2 = self._bias_variable([self._actions], name='fc2_bias')

        with tf.name_scope('Q'):
            qvalue = tf.matmul(h_fc1, w_fc2) + b_fc2

        return stateinput, qvalue, w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_fc2, b_fc2

    def _copy_target_qnetwork(self):
        self._session.run(self._copy_target_qnetwork_operation)

    def _train_qnetwork(self):
        # Step 1: obtain random minibatch from replay memory
        def __reshape_action(action):
            actions = np.zeros(self._actions)
            actions[action] = 1
            return actions

        # zip(*X) transposes X
        state_batch, \
        action_batch, \
        reward_batch, \
        next_state_batch, \
        terminal_batch = zip(*random.sample(self._replay_memory, self._minibatch_size))

        action_batch = [__reshape_action(action) for action in action_batch]

        # Step 2: calculate y
        y_batch = []
        qvalue_batch = self._qvalue_t.eval(feed_dict={self._stateinput_t:next_state_batch})
        for i in range(0, self._minibatch_size):
            terminal = terminal_batch[i]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self._gamma * np.max(qvalue_batch[i]))

        if self._timestep == self._observe_state_length + 1:
            run_metadata = tf.RunMetadata()
            self._session.run(self._train_step,
                              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                              run_metadata=run_metadata,
                              feed_dict={self._y_input : y_batch,
                                         self._action_input : action_batch,
                                         self._stateinput : state_batch})
            self._summarywriter.add_run_metadata(run_metadata, 'step_%d' % self._timestep)
        else:
            self._train_step.run(feed_dict={self._y_input : y_batch,
                                            self._action_input : action_batch,
                                            self._stateinput : state_batch})

        # calculate loss function (we do it via proxy variable since summaries are run separatly)
        value = self._cost.eval(feed_dict={self._y_input : y_batch,
                                           self._action_input : action_batch,
                                           self._stateinput : state_batch})
        self._loss_function.assign(value).op.run()

        # save network every 100000 iteration
        if self._timestep % 100000 == 0:
            self._saver.save(self._session, 'dqn', global_step=self._timestep)

        if self._timestep % self._q_target_update_interval == 0:
            self._copy_target_qnetwork()

    def _write_counters(self):
        # FPS
        new_timestamp = time.clock()
        self._fps.assign((self._timestep - self._prev_timestep) / (new_timestamp - self._prev_timestamp)).op.run()
        self._prev_timestamp = new_timestamp
        self._prev_timestep = self._timestep

        summary = self._session.run(self._summaries)
        self._summarywriter.add_summary(summary, self._timestep)


    def _weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)

    def _bias_variable(self, shape, name):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, name=name)

    def _conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

    def _max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
