This branch contains the DDPM code we used for the experiments in Section 5.1 of the [paper](https://arxiv.org/pdf/2405.12399).

```bash
git checkout ddpm
```

Assuming you have followed the installation steps on the [main branch](https://github.com/eloialonso/diamond/tree/main), you'll need to run in addition:

```bash
pip install diffusers==0.17.0
```

Note that the experiments in Section 5.1 trained the world models on a static dataset collected with an expert policy (use `training.model_free=True` to train a model-free expert, then collect a dataset and put its path into `collection.path_to_static_dataset`).

Otherwise, to train from scratch a DDPM-based diamond agent, run:

```bash
python src/main.py
```
