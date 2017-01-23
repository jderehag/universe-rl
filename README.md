[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

# Universe-rl: A high-level reinforcement learning library for OpenAi Universe

Universe-rl aims to provide a collection of different reinforcement learning 
algorithms ready to use out of the box with the 
[OpenAi Universe](https://universe.openai.com/) platform.

## Run an universe-rl agent locally

If you want to run the universe-rl agent locally on ubuntu 16.04 or higher you 
will need to install the following dependencies:

```
  sudo apt-get install swig python-opencv cmake zlib1g-dev golang libjpeg-turbo8-dev libgtk2.0-dev pkg-config
  git clone https://github.com/jderehag/universe-rl.git
  cd universe-rl
  pip install -r pip-requirements.txt
```

**Note**: if you installed python through [conda](https://www.continuum.io/downloads)
you might need to install opencv with the following command: 

```
  conda update conda
  conda install -c menpo opencv3
```

You can then run the agent with: `python run_game.py`. It should work with either 
python 2.7 or 3.5.

## Run an universe-rl agent through docker

You will need [docker](https://www.docker.com/) installed on your machine.

Before running your agent you will need to start the OpenAI Universe remote you
wish to connect to. To run the atari games provided by OpenAI the command is:

```
docker run -t -p 5900:5900 -p 15900:15900 quay.io/openai/universe.gym-core:latest
```

Once your remote is running you can start an agent either locally or with docker.

To run the default agent with docker simply do:

```
  docker run -t -p 6006:6006 --network host aurefranky/universe-rl
```

This will connect to remote and launch the default game [gym-core.Breakout-v0](https://universe.openai.com/envs/Breakout-v0). 
You can specify which game you want to connect to by passing the game name as 
an argument.

## View your Agent Playing

The OpenAi Universe Remote provides you with a way to view your agent playing 
the game. To do that you will need to navigate to 
[http://localhost:15900/viewer](http://localhost:15900/viewer) if you are running 
the remote locally.

## View your Agent Performance

You can monitor your agents performance through tensorboard, by default 
tensorboard will be available at [http://localhost:6006/](http://localhost:6006/).
