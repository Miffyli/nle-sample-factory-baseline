# NLE challenge baseline using sample-factory

An openly shared baseline code for the [NeurIPS 2021 Nethack challenge](https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/), using [sample-factory](https://github.com/alex-petrenko/sample-factory) in its core. Feel free to use in your submissions as you see fit! Note that there is no submission code. This code only works on Linux.

Core features:
- Trains two billion (2e9) steps in 22h on a single RTX 2080Ti and 16 2.3Ghz cores. This reaches an average of 750-800 reward.
- Learning algorithm is asynchronous PPO (see sample-factory for detailed explanation) with V-trace. Network consists of separate input heads and an RNN core (using GRUs).
- Main observation is an RGB image around the player character, rendered with `obs_wrappers.RenderCharImagesWithNumpyWrapper`, processed with a standard CNN used with Atari experiments.
- Agent also receives the `blstats` observation, normalized with manually set normalization weights, and the `message` observation. Both are processed with a two-layer network before the RNN. This does not do proper text processing for `message`, but at least allows it to detect common situations, e.g. "It is a wall". These encodings are concatenated with image encoding before the RNN core.

## Installation and running

Install requirements with `pip install -r requirements.txt`.

Run code with `./run_baseline.sh`. This should start printing out text about initializing the workers, and eventually learning statistics. Training lasts for two billion steps.

You can try to speed up training by changing the `num_workers` and `num_envs_per_worker` parameters inside `run_baseline.sh`.

## Contents

- `main.py`: entry point.
- `env.py`: core environment wrappers and creation of environment in sample-factory
- `obs_wrappers.py`: code for drawing RGB images of the NLE and processing `blstats` info.
- `models.py`: torch model for encoding observations before the RNN core.
- `run_baseline.sh`: launch the code with set hyperparameters.

## Wandb integration

By default the sample-factory stores logs as tensorboard files, but to ease up tracking, this code comes with Weights & Biases integration.

Simply define `WANDB_API_KEY` variable in the environment and install wandb (`pip install wandb`), and you should start seeing logs on the wandb page once you launch the code.

![wandb image](media/wandb.png?raw=true)