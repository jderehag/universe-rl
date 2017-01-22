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
from __future__ import print_function
import argparse
import os

import gym
import universe #pylint: disable=unused-import

from rl import dqn
from util import setup_logging
from util.render import LocalRenderer
from util.env_wrapper import create_env


def _play_game(args):
    env = create_env(args.game)

    agent = dqn.DQN(env.action_space.n, checkpointpath=args.checkpoints, summarypath=args.summary)
    window = LocalRenderer()

    action = 0
    while True:
        for frame, reward, terminal, _ in zip(*env.step((env.action_space.actions[action],))):
            if terminal:
                env.reset()

            if frame is not None:
                window.render(frame)
                action = agent.get_action(frame, action, reward, terminal)


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

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug output')

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.list_games:
        for game in sorted([env_spec.id for env_spec in gym.envs.registry.all()]):
            print(game)
    else:
        _play_game(args)


if __name__ == '__main__':
    _main()
