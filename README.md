# Diffusion for World Modeling: Visual Details Matter in Atari

**TL;DR** We introduce DIAMOND (DIffusion As a Model Of eNvironment Dreams), a reinforcement learning agent trained in a diffusion world model.

>[Install](#installation), then try our [pretrained world models](#try)!
>
>```bash
>python src/play.py --pretrained
>```

<div align='center'>
  Autoregressive imagination with DIAMOND on a subset of Atari games
  <img alt="DIAMOND's world model in Breakout, Pong, KungFuMaster, Boxing, Asterix" src="assets/main.gif">
</div>

<a name="quick_links"></a>
## Quick Links

- [Installation](#installation)
- [Try our playable diffusion world models](#try)
- [Launch a training run](#launch)
- [Configuration](#configuration)
- [Visualization](#visualization)
  - [Play mode (default)](#play_mode)
  - [Dataset mode (add `-d`)](#dataset_mode)
  - [Other options, common to play/dataset modes](#other_options)
- [Run folder structure](#structure)
- [Results](#results)
- [Credits](#credits)


<a name="installation"></a>
## [⬆️](#quick_links) Installation

Download the repository:

```bash
mkdir diamond
cd diamond
wget -c https://anonymous.4open.science/api/repo/_diamond/zip -O tmp.zip
# curl https://anonymous.4open.science/api/repo/_diamond/zip -o tmp.zip   # if you do not have wget but do have curl 
unzip tmp.zip
rm tmp.zip
```

We recommend using [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) to create a new environment:

```bash
conda create -n diamond python=3.10
conda activate diamond
```

Install dependencies listed in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

**Warning**: Atari ROMs will be downloaded with the dependencies, which means that you acknowledge that you have the license to use them.

<a name="try"></a>
## [⬆️](#quick_links) Try our playable diffusion world models

```bash
python src/play.py --pretrained
```

Then select a game, and world model and policy pretrained on Atari 100k will be downloaded from our [repository on Hugging Face Hub 🤗](https://huggingface.co/zLPwHqz4cu6JkNUY/diamond) and cached on your machine.

First things you might want to try:
- Press `m` to take control (the policy is playing by default).
- Press `↑` to increase the imagination horizon (default is 15, which is frustrating when playing yourself).

To adjust the sampling parameters (number of denoising steps, stochasticity, order, etc) of the trained diffusion world model, for instance to trade off sampling speed and quality, edit the section `world_model_env.diffusion_sampler` in the file `config/trainer.yaml`.

See [Visualization](#visualization) for more details about the available commands and options.

<a name="launch"></a>
## [⬆️](#quick_links) Launch a training run

To train with the hyperparameters used in the paper, launch:
```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0
```

This creates a new folder for your run, located in `outputs/YYYY-MM-DD/hh-mm-ss/`.

To resume a run that crashed, navigate to the fun folder and launch:

```bash
./scripts/resume.sh
```

<a name="configuration"></a>
## [⬆️](#quick_links) Configuration

We use [Hydra](https://github.com/facebookresearch/hydra) for configuration management.

All configuration files are located in the `config` folder:

- `config/trainer.yaml`: main configuration file.
- `config/agent/default.yaml`: architecture hyperparameters.
- `config/env/atari.yaml`: environment hyperparameters.

You can turn on logging to [weights & biases](https://wandb.ai) in the `wandb` section of `config/trainer.yaml`.

Set `training.model_free=true` in the file `config/trainer.yaml` to "unplug" the world model and perform standard model-free reinforcement learning.

<a name="visualization"></a>
## [⬆️](#quick_links) Visualization

<a name="play_mode"></a>
### [⬆️](#quick_links) Play mode (default)

To visualize your last checkpoint, launch **from the run folder**:

```bash
python src/play.py
```

By default, you visualize the policy playing in the world model. To play yourself, or switch to the real environment, use the controls described below.

```txt
Controls (play mode)

(Game-specific commands will be printed on start up)

⏎   : reset environment

m   : switch controller (policy/human)
↑/↓ : imagination horizon (+1/-1)
←/→ : next environment [world model ←→ real env (test) ←→ real env (train)]

.   : pause/unpause
e   : step-by-step (when paused)
```

Add `-r` to toggle "recording mode" (works only in play mode). Every completed episode will be saved in `dataset/rec_<env_name>_<controller>`. For instance:

- `dataset/rec_wm_π`: Policy playing in world model.
- `dataset/rec_wm_H`: Human playing in world model.
- `dataset/rec_test_H`: Human playing in test real environment.

You can then use the "dataset mode" described in the next section to replay the stored episodes.

<a name="dataset_mode"></a>
### [⬆️](#quick_links) Dataset mode (add `-d`)

**In the run folder**, to visualize the datasets contained in the `dataset` subfolder, add `-d` to switch to "dataset mode":

```bash
python src/play.py -d
```

You can use the controls described below to navigate the datasets and episodes.

```txt
Controls (dataset mode)

m   : next dataset (if multiple datasets, like recordings, etc) 
↑/↓ : next/previous episode
←/→ : next/previous timestep in episodes
PgUp: +10 timesteps
PgDn: -10 timesteps
⏎   : back to first timestep
```

<a name="other_options"></a>
### [⬆️](#quick_links) Other options, common to play/dataset modes

```txt
--fps FPS             Target frame rate (default 15).
--size SIZE           Window size (default 800).
--no-header           Remove header.
```

<a name="structure"></a>
## [⬆️](#quick_links) Run folder structure

Each new run is located at `outputs/YYYY-MM-DD/hh-mm-ss/`. This folder is structured as follows:

```txt
outputs/YYYY-MM-DD/hh-mm-ss/
│
└─── checkpoints
│   │   state.pt  # full training state
│   │
│   └─── agent_versions
│       │   ...
│       │   agent_epoch_00999.pt
│       │   agent_epoch_01000.pt  # agent weights only
│
└─── config
│   |   trainer.yaml
|
└─── dataset
│   │
│   └─── train
│   |   │   info.pt
│   |   │   ...
|   |
│   └─── test
│       │   info.pt
│       │   ...
│
└─── scripts
│   │   resume.sh
|   |   ...
|
└─── src
|   |   main.py
|   |   ...
|
└─── wandb
    |   ...
```

<a name="results"></a>
## [⬆️](#quick_links) Results

The file [results/data/DIAMOND.json](results/data/DIAMOND.json) contains the results for each game and seed used in the paper.

<a name="credits"></a>
## [⬆️](#quick_links) Credits

- [https://github.com/crowsonkb/k-diffusion/](https://github.com/crowsonkb/k-diffusion/)
- [https://github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
