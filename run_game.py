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

def _preprocess(observation):
    # preprocess raw image to 80*80 gray image
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    return cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)[1]

def _play_game(args):
    env = wrappers.SafeActionSpace(gym.make(args.game))
    env.configure(remotes=1)
    env.reset()

    agent = dqn.DQN(len(env.action_space.actions), checkpointpath=args.checkpoints, summarypath=args.summary)

    action = 0
    while True:
        # For now, assume single instance environment
        frame, reward, terminal, _ = env.step([env.action_space[action]])
        assert len(frame) == 1
        action = agent.get_action(_preprocess(frame[0]), action, reward, terminal)
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
    parser.add_argument('game', nargs='?', default='flashgames.DuskDrive-v0',
                        help="The openai.universe game-id to run")
    args = parser.parse_args()

    if args.list_games:
        for game in sorted([env_spec.id for env_spec in gym.envs.registry.all()]):
            print game

    else:
        _play_game(args)

if __name__ == '__main__':
    _main()
