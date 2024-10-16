# Diffusion for World Modeling: Visual Details Matter in Atari (NeurIPS 2024 Spotlight)

This branch contains the code to play (and train) our world model of *Counter-Strike: Global Offensive* (CS:GO).

🌍 [Project Page](https://diamond-wm.github.io) • 🤓 [Paper](https://arxiv.org/pdf/2405.12399) • 𝕏 [Atari thread](https://x.com/EloiAlonso1/status/1793916382779982120) • 𝕏 [CSGO thread](https://x.com/EloiAlonso1/status/1844803606064611771) • 💬 [Discord](https://discord.gg/74vha5RWPg)

<div align='center'>
  Human player in CSGO world model (full quality video <a href="https://diamond-wm.github.io/static/videos/grid.mp4">here</a>)
  <br>
  <img alt="DIAMOND agent in WM" src="https://github.com/user-attachments/assets/dcbdd523-ca22-46a9-bb7d-bcc52080fe00">
</div>

## Installation
```bash
git clone https://github.com/eloialonso/diamond.git
cd diamond
git checkout csgo
conda create -n diamond python=3.10
conda activate diamond
pip install -r requirements.txt
python src/play.py
```

> Note on Apple Silicon you must enable CPU fallback for [MPS backend](https://pytorch.org/docs/stable/notes/mps.html) with
> `PYTORCH_ENABLE_MPS_FALLBACK=1 python src/play.py`

The final command will automatically download our trained CSGO diffusion world model from the [HuggingFace Hub 🤗](https://huggingface.co/eloialonso/diamond/tree/main) along with spawn points and human player actions. Note that the model weights require 1.5GB of disk space.

When the download is complete, control actions will be printed in the terminal. Press Enter to start playing.

The default [fast config](config/world_model_env/fast.yaml) runs best on a machine with a CUDA GPU, but can also be run on CPU at reduced fps. The model also runs faster if compiled (but takes longer at startup).
```bash
python src/play.py --compile
```

To reproduce our videos, you can change the [trainer](config/trainer.yaml#L5) file to use the [higher_quality](config/world_model_env/higher_quality.yaml) config (instead of the [fast](config/world_model_env/fast.yaml) config) with increased denoising steps to enable higher quality generation at reduced speed (10fps on a RTX 3090 on our machine).

To adjust the sampling parameters yourself (number of denoising steps, stochasticity, order, etc) of the trained diffusion world model, for instance to trade off sampling speed and quality, edit the file `config/world_model_env/fast.yaml`.

## Training

**IMPORTANT**: Any issue related to the download of training data should be reported on the [dataset repo](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning).

We trained on the biggest dataset of the repo (`dataset_dm_scraped_dust2`), that corresponds to 5.5M frames, 95h of gameplay, and takes ~660Gb of disk space.

- We used a random test split of 500 episodes of 1000 steps (specified in [test_split.txt](test_split.txt)).
- We used the remaining 5003 episodes to train the model. This corresponds to 5M frames, or 87h of gameplay.

To get the data ready for training on your machine:
- **Step 1**: Download the `.tar` files from the `dataset_dm_scraped_dust2_tars` on the [OneDrive link](https://1drv.ms/u/s!AjG1JlThUkPgh1JEIxETxvaphzgC?e=2AJfA3) from the [dataset repo](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning). 
- **Step 2**: Use our [script](src/process_csgo_tar_files.py) to prepare the downloaded data for training, as follows:

```bash
python src/process_csgo_tar_files.py <folder_with_tar_files_from_step_one> <folder_to_store_processed_data>
```

Then edit [config/env/csgo.yaml](config/env/csgo.yaml) and set:
- `path_data_low_res` to `<folder_to_store_processed_data>/low_res`
- `path_data_full_res` to `<folder_to_store_processed_data>/full_res`

You can then launch a training run with `python src/main.py`.

The provided configuration took 12 days on a RTX 4090.

---

<a name="citation"></a>
## [⬆️](#quick-links) Citation

```text
@inproceedings{alonso2024diffusionworldmodelingvisual,
      title={Diffusion for World Modeling: Visual Details Matter in Atari},
      author={Eloi Alonso and Adam Jelley and Vincent Micheli and Anssi Kanervisto and Amos Storkey and Tim Pearce and François Fleuret},
      booktitle={Thirty-eighth Conference on Neural Information Processing Systems}}
      year={2024},
      url={https://arxiv.org/abs/2405.12399},
}
```

<a name="credits"></a>
## [⬆️](#quick_links) Credits

- [https://github.com/crowsonkb/k-diffusion/](https://github.com/crowsonkb/k-diffusion/)
- [https://github.com/huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)
- [https://github.com/google-research/rliable](https://github.com/google-research/rliable)
- [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- [https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/)
