import argparse
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset
from envs import make_atari_env, WorldModelEnv
from game import ActionNames, DatasetEnv, Game, get_keymap_and_action_names, Keymap, NamedEnv, PlayEnv
from utils import get_path_agent_ckpt, prompt_atari_game


OmegaConf.register_new_resolver("eval", eval)


def download(filename: str) -> Path:
    path = hf_hub_download(repo_id="zLPwHqz4cu6JkNUY/diamond", filename=filename)
    return Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained", action="store_true", help="Download pretrained world model and agent.")
    parser.add_argument("-d", "--dataset-mode", action="store_true", help="Dataset visualization mode.")
    parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
    parser.add_argument("-n", "--num-steps-initial-collect", type=int, default=1000, help="Num steps initial collect.")
    parser.add_argument("-c", "--pick-checkpoint", action="store_true", help="Display a list of checkpoints to choose from.")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate.")
    parser.add_argument("--size", type=int, default=640, help="Window size.")
    parser.add_argument("--no-header", action="store_true")
    return parser.parse_args()


def prepare_dataset_mode(cfg: DictConfig) -> Tuple[DatasetEnv, Keymap, ActionNames]:
    datasets = []
    for p in Path("dataset").iterdir():
        if p.is_dir():
            d = Dataset(p, p.stem)
            d.load_from_default_path()
            datasets.append(d)
    _, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    dataset_env = DatasetEnv(datasets, env_action_names)
    keymap, _ = get_keymap_and_action_names("dataset_mode")
    return dataset_env, keymap


def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace) -> Tuple[PlayEnv, Keymap, ActionNames]:
    # Checkpoint
    if args.pretrained:
        if args.pick_checkpoint:
            print("Warning: --pretrained is on, ignoring --pick-checkpoint")
        name = prompt_atari_game()
        path_ckpt = download(f"{name}.pt")
        
        # Override config
        cfg.agent = OmegaConf.load(download("default_agent_config.yaml"))
        cfg.env = OmegaConf.load(download("default_atari_config.yaml"))
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
    else:
        epoch = int(input("Enter the checkpoint you want to load: epoch=")) if args.pick_checkpoint else -1
        path_ckpt = get_path_agent_ckpt("checkpoints", epoch=epoch)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Real envs
    train_env = make_atari_env(num_envs=1, device=device, **cfg.env.train)
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    agent.load(path_ckpt)

    # Collect for imagination's initialization
    n = args.num_steps_initial_collect
    dataset = Dataset(Path(f"dataset/{path_ckpt.stem}_{n}"))
    dataset.load_from_default_path()
    if len(dataset) == 0:
        print(f"Collecting {n} steps in real environment for world model initialization.")
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
        collector.send(NumToCollect(steps=n))
        dataset.save_to_default_path()

    # World model environment
    bs = BatchSampler(dataset, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)

    envs = [
        NamedEnv("wm", wm_env),
        NamedEnv("test", test_env),
        NamedEnv("train", train_env),
    ]

    env_keymap, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    play_env = PlayEnv(agent, envs, env_action_names, env_keymap, args.record)

    return play_env, env_keymap


@torch.no_grad()
def main():
    args = parse_args()
    if args.dataset_mode:
        if not Path("dataset").is_dir():
            print(f"Error: {str(Path('dataset').absolute())} not found, cannot use dataset mode.")
            return
        if Path(".git").is_dir():
            print(f"Error: cannot run dataset mode the root of the repository.")
            return
        if args.pretrained or args.record:
            print("Warning: dataset mode, ignoring --pretrained and --record")

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    env, keymap = prepare_dataset_mode(cfg) if args.dataset_mode else prepare_play_mode(cfg, args)
    size = (args.size // cfg.env.train.size) * cfg.env.train.size  # window size
    game = Game(env, keymap, (size, size), fps=args.fps, verbose=not args.no_header)
    game.run()


if __name__ == "__main__":
    main()
