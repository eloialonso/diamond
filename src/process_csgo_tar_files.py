"""
IMPORTANT:
This script prepares data downloaded from the OneDrive link provided on the repo that introduced the dataset: https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
=> Any issue related to the download of this data should be reported on the dataset repo linked above (NOT on DIAMOND's repo)

This script should be called with exactly 2 positional arguments:

- <tar_dir>: folder containing the .tar files from `dataset_dm_scraped_dust2_tars` folder on the OneDrive
- <out_dir>: a new dir (should not exist already), the script will untar and process data there
"""

import argparse
from functools import partial
from pathlib import Path
from multiprocessing import Pool
import shutil
import subprocess

import torch
import torchvision.transforms.functional as T
from tqdm import tqdm

from data.dataset import Dataset, CSGOHdf5Dataset
from data.episode import Episode
from data.segment import SegmentId


PREFIX = "hdf5_dm_july2021_"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tar_dir",
        type=Path,
        help="folder containing the .tar files from `dataset_dm_scraped_dust2_tars` folder on the OneDrive",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        help="a new directory (should not exist already), the script will untar and process data there",
    )
    return parser.parse_args()


def process_tar(path_tar: Path, out_dir: Path, remove_tar: bool) -> None:
    d = path_tar.stem
    assert path_tar.stem.startswith(PREFIX)
    d = out_dir / "-".join(path_tar.stem[len(PREFIX) :].split("_to_"))
    d.mkdir(exist_ok=False, parents=True)
    shutil.move(path_tar, d)
    subprocess.run(f"cd {d} && tar -xvf {path_tar.name}", shell=True)
    new_path_tar = d / path_tar.name
    if remove_tar:
        new_path_tar.unlink()
    else:
        shutil.move(new_path_tar, path_tar.parent)


def main():
    args = parse_args()

    tar_dir = args.tar_dir.absolute()
    out_dir = args.out_dir.absolute()

    if not tar_dir.exists():
        print(
            "Wrong usage: the tar directory should exist (and contain the downloaded .tar files)"
        )
        return

    if out_dir.exists():
        print(f"Wrong usage: the output directory should not exist ({args.out_dir})")
        return

    with Path("test_split.txt").open("r") as f:
        test_files = f.read().split("\n")

    full_res_dir = out_dir / "full_res"
    low_res_dir = out_dir / "low_res"

    tar_files = [
        x for x in tar_dir.iterdir() if x.suffix == ".tar" and x.stem.startswith(PREFIX)
    ]
    n = len(tar_files)

    if (
        n < 28
        and input(
            f"Found only {n} .tar files instead of 28, so it looks like the data has not been entirely downloaded, continue? [y|N] "
        )
        != "y"
    ):
        return

    str_files = "\n".join(map(str, tar_files))
    print(f"Ready to untar {n} tar files:\n{str_files}")

    remove_tar = (
        input("Remove .tar files once they are processed? [y|N] ").lower() == "y"
    )

    # Untar CSGO files
    f = partial(process_tar, out_dir=full_res_dir, remove_tar=remove_tar)
    with Pool(n) as p:
        p.map(f, tar_files)

    print(f"{n} .tar files unpacked in {full_res_dir}")

    #
    # Create low-res data
    #

    csgo_dataset = CSGOHdf5Dataset(full_res_dir)

    train_dataset = Dataset(low_res_dir / "train", None)
    test_dataset = Dataset(low_res_dir / "test", None)

    for i in tqdm(csgo_dataset._filenames, desc="Creating low_res"):
        episode = Episode(
            **{
                k: v
                for k, v in csgo_dataset[SegmentId(i, 0, 1000)].__dict__.items()
                if k not in ("mask_padding", "id")
            }
        )
        episode.obs = T.resize(
            episode.obs, (30, 56), interpolation=T.InterpolationMode.BICUBIC
        )
        filename = csgo_dataset._filenames[i]
        file_id = f"{filename.parent.stem}/{filename.name}"
        episode.info = {"original_file_id": file_id}
        dataset = test_dataset if filename.name in test_files else train_dataset
        dataset.add_episode(episode)

    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()

    print(
        f"Split train/test data ({train_dataset.num_episodes}/{test_dataset.num_episodes} episodes)\n"
    )

    print("You can now edit `config/env/csgo.yaml` and set:")
    print(f"path_data_low_res: {low_res_dir}")
    print(f"path_data_full_res: {full_res_dir}")


if __name__ == "__main__":
    main()
