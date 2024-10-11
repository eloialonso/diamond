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

The [default config](config/world_model_env/default.yaml) runs best on a machine with a CUDA GPU. The model also runs faster if compiled (but takes longer at startup).
```bash
python src/play.py --compile
```

If you only have CPU or you would like the world model to run faster, you can change the [trainer](config/trainer.yaml#L5) file to use the [fast](config/world_model_env/fast.yaml) config (instead of the [default](config/world_model_env/default.yaml)) with reduced denoising steps to enable faster generation at lower quality.

To adjust the sampling parameters yourself (number of denoising steps, stochasticity, order, etc) of the trained diffusion world model, for instance to trade off sampling speed and quality, edit the file `config/world_model_env/default.yaml`.

## Data

For training we used the dataset `dataset_dm_scraped_dust2` from [Counter-Strike Deathmatch with Large-Scale Behavioural Cloning](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/).

We used a random test split of 500 episodes of 1000 steps (see [test_split.txt](test_split.txt)).

We used the remaining 5003 episodes to train the model. This corresponds to 5M frames, or 87h of gameplay.

## Training time

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
