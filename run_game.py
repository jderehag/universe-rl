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
import argparse

import cv2
import gym
import universe  # pylint: disable=unused-import
from universe import wrappers

from rl import dqn

class GenericActionSpace(object):
    '''
    Represents a translatable action space across many different environments
    This is needed due to that universe action_spaces are not API compatible and needs to be slightly modified for
    every enevironment.
    '''
    def __init__(self, action_space):
        self.action_space = action_space
        if isinstance(action_space, universe.spaces.hardcoded.Hardcoded):
            self.n = len(self.action_space.actions)
            self.actions = self.action_space.actions
        else:
            self.n = self.action_space.n
            self.actions = range(0, self.n)

    def __getitem__(self, i):
        return self.actions[i]

def _make_generic_frame(frame):
    if isinstance(frame, dict):
        return frame['vision']

    return frame

def _preprocess(observation):
    # preprocess raw image to 80*80 gray image
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    return cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)[1]

def _play_game(args):
    env = wrappers.SafeActionSpace(gym.make(args.game))
    env.configure()
    env.reset()
    action_space = GenericActionSpace(env.action_space)

    agent = dqn.DQN(action_space.n, checkpointpath=args.checkpoints, summarypath=args.summary)

    action = 0
    while True:
        # For now, assume single instance environment (1 frame/action per observation)
        # zip(*X) Transposes observations
        for frame, reward, terminal, _ in zip(*env.step([action_space[action]])):
            if frame is not None:
                action = agent.get_action(_preprocess(_make_generic_frame(frame)), action, reward, terminal)
                env.render()

def _main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    data_basepath = 'data'
    parser.add_argument('-c', '--checkpoints', dest='checkpoints', default=os.path.join(data_basepath, 'checkpoints'),
                        help='Path where to store checkpoints (i.e partial training)')
    parser.add_argument('-s', '--summary', dest='summary', default=os.path.join(data_basepath, 'summary'),
                        help='Path where to store summary data (for tensorboard)')
    parser.add_argument('-l', '--list', dest='list_games', action='store_true', default=False,
                        help="List all available openai.universe games")
    parser.add_argument('game', nargs='?', default='Breakout-v0',
                        help="The openai.universe game-id to run")
    args = parser.parse_args()

    if args.list_games:
        for game in sorted([env_spec.id for env_spec in gym.envs.registry.all()]):
            print game

    else:
        _play_game(args)

if __name__ == '__main__':
    _main()
