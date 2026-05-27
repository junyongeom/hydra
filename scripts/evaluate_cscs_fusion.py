#!/usr/bin/env python
"""Evaluate proposed HYDRA CSC&S Fusion checkpoints scene by scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def resolve_device(value: str):
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_data_root(root_path: str) -> str:
    path = Path(root_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not (path / "Scene1").is_dir() and (path / "XRF55" / "Scene1").is_dir():
        path = path / "XRF55"
    return str(path)


def checkpoint_work_dir(checkpoint_dir: str | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    path = Path(checkpoint_dir).expanduser().resolve()
    return path.parent if path.name == "reproc" else path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("XRF55_ROOT", "data/XRF55"))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shot", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--ckpt-pattern",
        default="reproc/CSCS_Fusion_{scene}_{shot}shot_ft.pth",
        help="Checkpoint pattern relative to checkpoint work dir.",
    )
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "The proposed HYDRA Fusion model requires CUDA because it uses "
            "mamba_ssm/causal-conv1d CUDA kernels. Pass --device cuda on a CUDA machine."
        )

    root_path = normalize_data_root(args.data_root)
    os.environ["XRF55_ROOT"] = root_path

    work_dir = checkpoint_work_dir(args.checkpoint_dir)
    if work_dir is not None:
        os.chdir(work_dir)

    from hydra.experiments.proposed_hydra import (
        FusionClassifier,
        build_cscs_eval_loaders,
        evaluate_model,
    )

    loaders = build_cscs_eval_loaders(
        root_path,
        BS=args.batch_size,
        NUM_WORKERS=args.num_workers,
        shot=args.shot,
    )

    results = {}
    for scene in ["Scene2", "Scene3", "Scene4"]:
        ckpt = args.ckpt_pattern.format(scene=scene, shot=args.shot)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        model = FusionClassifier(num_classes=55).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        acc = evaluate_model(model, loaders[scene], device)
        results[scene] = acc
        print(f"{scene}: {acc:.2f}%  ({ckpt})")
        del model
        torch.cuda.empty_cache()

    avg = sum(results.values()) / len(results)
    print(f"avg: {avg:.2f}%")

    if args.save_json:
        payload = {
            "task": "cscs",
            "model": "proposed_hydra_fusion",
            "shot": args.shot,
            "ckpt_pattern": args.ckpt_pattern,
            "results": results,
            "avg": avg,
        }
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
