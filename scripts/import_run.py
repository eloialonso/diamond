#! /usr/bin/env python

import argparse
from functools import partial
import json
from pathlib import Path
import subprocess
from typing import Optional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--user", type=Optional[str])
    parser.add_argument("--rootdir", type=Optional[str])
    args = parser.parse_args()

    run = partial(subprocess.run, shell=True, check=True, text=True)
    host = args.host if args.user is None else f"{args.user}@{args.host}"

    def run_remote_cmd(cmd):
        return subprocess.check_output(f"ssh {host} {cmd}", shell=True, text=True)

    def ls(p):
        out = run_remote_cmd(f"ls {p}")
        return out.strip().split("\n")[::-1]

    def ask(l, info=None):
        print(
            "\n".join(
                [
                    f"{i:{len(str(len(l)))}d}: {d}"
                    + (f" ({info[d]})" if info is not None else "")
                    for i, d in enumerate(l, 1)
                ]
            )
        )
        while True:
            i = input("\nEnter a number: ")
            if i.isdigit() and 1 <= int(i) <= len(l):
                break
            print("\n/!\\ Invalid choice\n")
        return l[int(i) - 1]

    def ask_if_verbose(question, default):
        if not args.verbose:
            return default
        suffix = "[Y|n]" if default else "[y|N]"
        answer = input(f"{question} {suffix} ").lower()

        return (answer != "n") if default else (answer == "y")

    def get_info(rundir):
        return json.loads(
            run_remote_cmd(f"cat {rundir}/checkpoints/info_for_import_script.json")
        )

    if args.rootdir is None:
        for p in Path(__file__).resolve().parents:
            if (p / ".git").is_dir():
                break
        else:
            raise RuntimeError("This file is not in a git repository")
        out = run_remote_cmd(f"find -type d -name {p.name}").strip().split("\n")
        assert len(out) == 1
        rootdir = out[0]
    else:
        rootdir = f'{args.rootdir.strip().strip("/")}'

    dates = ls(f"{rootdir}/outputs")
    date = ask(dates)
    times = ls(f"{rootdir}/outputs/{date}")

    infos = {
        time: get_info(rundir=f"{rootdir}/outputs/{date}/{time}") for time in times
    }
    time = ask(times, infos)

    src = f"{rootdir}/outputs/{date}/{time}"

    dst = Path(args.host) / date
    dst.mkdir(exist_ok=True, parents=True)

    exclude = [
        "*.log",
        "checkpoints/*",
        "checkpoints_tmp",
        ".hydra",
        "media",
        "__pycache__",
        "wandb",
    ]

    include = ["checkpoints/agent_versions"]

    if ask_if_verbose("Download only last checkpoint?", default=True):
        last_ckpt = ls(f"{src}/checkpoints/agent_versions")[0]
        exclude.append("checkpoints/agent_versions/*")
        include.append(f"checkpoints/agent_versions/{last_ckpt}")

    if not ask_if_verbose("Download train dataset?", default=False):
        exclude.append("dataset/train")

    if not ask_if_verbose("Download test dataset?", default=False):
        exclude.append("dataset/test")

    cmd = "rsync -av"
    for i in include:
        cmd += f' --include="{i}"'
    for e in exclude:
        cmd += f' --exclude="{e}"'

    cmd += f" {host}:{src} {str(dst)}"
    run(cmd)

    path = (dst / time).absolute()
    print(f"\n--> Run imported in:\n{path}")
    run(f"echo {path} | xclip")


if __name__ == "__main__":
    main()
