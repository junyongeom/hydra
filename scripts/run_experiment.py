#!/usr/bin/env python
"""Run a configured HYDRA experiment."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


MODULES = {
    "proposed": "hydra.experiments.proposed_hydra",
    "proposed_hydra": "hydra.experiments.proposed_hydra",
    "xrf55": "hydra.experiments.xrf55_baseline",
    "xrf55_baseline": "hydra.experiments.xrf55_baseline",
    "zhang": "hydra.experiments.zhang_baseline",
    "zhang_baseline": "hydra.experiments.zhang_baseline",
}

RUNNERS = {
    ("proposed", "in_domain"): "run_in_domain_scene1",
    ("proposed", "cross_subject"): "run_cross_subject_21_9",
    ("proposed", "cross_scene"): "run_cross_scene",
    ("proposed", "cscs"): "run_cscs",
    ("proposed", "ablation"): "run_cscs_ablation",
    ("proposed", "sensor_failure"): "run_sensor_failure",
    ("proposed", "complexity"): "measure_model_cost",
    ("xrf55", "in_domain"): "run_in_domain_scene1_xrf55",
    ("xrf55", "cross_subject"): "run_cross_subject_21_9_xrf55",
    ("xrf55", "cross_scene"): "run_cross_scene_xrf55",
    ("xrf55", "cscs"): "run_cscs_xrf55",
    ("xrf55", "sensor_failure"): "run_sensor_failure_xrf55",
    ("xrf55", "complexity"): "measure_model_cost",
    ("zhang", "in_domain"): "run_in_domain_scene1_zhang",
    ("zhang", "cross_subject"): "run_cross_subject_21_9_zhang",
    ("zhang", "cross_scene"): "run_cross_scene_zhang",
    ("zhang", "cscs"): "run_cscs_zhang",
    ("zhang", "sensor_failure"): "run_sensor_failure_zhang",
    ("zhang", "complexity"): "measure_model_cost",
}

TASK_KEYS = {
    "in_domain": [
        "root_path",
        "device",
        "epochs",
        "BS",
        "NUM_WORKERS",
        "modalities",
        "force_retrain",
    ],
    "cross_subject": [
        "root_path",
        "device",
        "pretrain_epochs",
        "ft_epochs",
        "ft_lr",
        "BS",
        "NUM_WORKERS",
        "modalities",
        "shots",
        "force_pretrain",
    ],
    "cross_scene": [
        "root_path",
        "device",
        "ft_epochs",
        "ft_lr",
        "BS",
        "NUM_WORKERS",
        "modalities",
        "shots",
        "strict_subject_consistency",
    ],
    "cscs": [
        "root_path",
        "device",
        "pretrain_epochs",
        "ft_epochs",
        "ft_lr",
        "BS",
        "NUM_WORKERS",
        "modalities",
        "shots",
        "force_pretrain",
    ],
    "ablation": [
        "tags",
        "root_path",
        "device",
        "pretrain_epochs",
        "ft_epochs",
        "ft_lr",
        "BS",
        "NUM_WORKERS",
        "shots",
        "force_pretrain",
    ],
    "sensor_failure": [
        "root_path",
        "device",
        "BS",
        "NUM_WORKERS",
        "shot",
        "tags",
    ],
    "complexity": [
        "device",
        "use_cache",
        "save",
        "verbose",
    ],
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def canonical_experiment(name: str) -> str:
    if name in {"proposed", "proposed_hydra"}:
        return "proposed"
    if name in {"xrf55", "xrf55_baseline"}:
        return "xrf55"
    if name in {"zhang", "zhang_baseline"}:
        return "zhang"
    raise ValueError(f"Unknown experiment: {name}")


def resolve_device(value: str):
    if value != "auto":
        return value
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_params(config: dict[str, Any], data_root: str | None, device_arg: str | None):
    training = config.get("training", {}) or {}
    ablation = config.get("ablation", {}) or {}
    sensor_failure = config.get("sensor_failure", {}) or {}
    root_path = data_root or config.get("data_root") or os.environ.get("XRF55_ROOT") or "data/XRF55"
    device = resolve_device(device_arg or config.get("device", "auto"))

    params = {
        "root_path": root_path,
        "device": device,
        "epochs": training.get("epochs", 100),
        "pretrain_epochs": training.get("pretrain_epochs", 100),
        "ft_epochs": training.get("ft_epochs", 100),
        "ft_lr": training.get("ft_lr", 1e-5),
        "BS": training.get("batch_size", 16),
        "NUM_WORKERS": training.get("num_workers", 4),
        "modalities": tuple(training.get("modalities", ["RFID", "WiFi", "mmWave", "Fusion"])),
        "shots": tuple(training.get("shots", [0, 1, 2, 3, 4, 5])),
        "shot": sensor_failure.get("shot", 5),
        "force_retrain": bool(training.get("force_retrain", False)),
        "force_pretrain": bool(training.get("force_pretrain", False)),
        "strict_subject_consistency": bool(training.get("strict_subject_consistency", True)),
        "tags": tuple(sensor_failure.get("tags") or ablation.get("tags", [])) or None,
        "use_cache": bool((config.get("complexity", {}) or {}).get("use_cache", True)),
        "save": bool((config.get("complexity", {}) or {}).get("save", True)),
        "verbose": bool((config.get("complexity", {}) or {}).get("verbose", True)),
    }
    return params


def checkpoint_work_dir(checkpoint_dir: str | None) -> Path | None:
    if checkpoint_dir is None:
        return None
    path = Path(checkpoint_dir).expanduser().resolve()
    return path.parent if path.name == "reproc" else path


def normalize_data_root(root_path: str) -> str:
    path = Path(root_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not (path / "Scene1").is_dir() and (path / "XRF55" / "Scene1").is_dir():
        path = path / "XRF55"
    return str(path)


def ensure_runtime_supported(experiment: str, device: Any) -> None:
    if experiment != "proposed":
        return
    import torch

    if torch.device(device).type != "cuda":
        raise RuntimeError(
            "The proposed HYDRA model uses mamba_ssm/causal-conv1d CUDA "
            "kernels and requires --device cuda or device: auto with CUDA "
            "available. Use the xrf55 or zhang baselines for CPU-only runs."
        )


def parse_int_list(value: str | None):
    if not value:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_str_list(value: str | None):
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def call_runner(fn, task: str, params: dict[str, Any]):
    selected = {key: params[key] for key in TASK_KEYS[task] if key in params}
    signature = inspect.signature(fn)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_kwargs:
        return fn(**selected)
    filtered = {key: val for key, val in selected.items() if key in signature.parameters}
    return fn(**filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", choices=sorted(MODULES), default=None)
    parser.add_argument("--task", choices=sorted(TASK_KEYS), default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--shots", default=None, help="Comma-separated shot list, e.g. 5 or 0,1,2,3,4,5.")
    parser.add_argument("--shot", type=int, default=None, help="Single shot value for sensor-failure evaluation.")
    parser.add_argument("--modalities", default=None, help="Comma-separated modalities, e.g. Fusion or RFID,WiFi,mmWave,Fusion.")
    parser.add_argument("--tags", default=None, help="Comma-separated ablation/sensor-failure tags, e.g. base,comp_drop.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Path to a reproc directory, or to a directory containing reproc/. "
            "The runner changes into the parent so reproc/*.pth paths resolve."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = canonical_experiment(args.experiment or config.get("experiment", "proposed"))
    task = args.task or config.get("task", "cscs")
    if (experiment, task) not in RUNNERS:
        raise ValueError(f"No runner registered for experiment={experiment!r}, task={task!r}")

    params = build_params(config, args.data_root, args.device)
    if task == "ablation":
        ablation = config.get("ablation", {}) or {}
        if "shots" in ablation:
            params["shots"] = tuple(ablation["shots"])
        elif "shot" in ablation:
            params["shots"] = (ablation["shot"],)
    elif task == "sensor_failure":
        sensor_failure = config.get("sensor_failure", {}) or {}
        if "shot" in sensor_failure:
            params["shot"] = sensor_failure["shot"]
        if "tags" in sensor_failure:
            params["tags"] = tuple(sensor_failure["tags"])

    shot_override = parse_int_list(args.shots)
    if shot_override is not None:
        params["shots"] = shot_override
    if args.shot is not None:
        params["shot"] = args.shot
    modality_override = parse_str_list(args.modalities)
    if modality_override is not None:
        params["modalities"] = modality_override
    tag_override = parse_str_list(args.tags)
    if tag_override is not None:
        params["tags"] = tag_override
    params["root_path"] = normalize_data_root(str(params["root_path"]))
    ensure_runtime_supported(experiment, params["device"])
    os.environ["XRF55_ROOT"] = str(params["root_path"])

    work_dir = checkpoint_work_dir(args.checkpoint_dir)
    if work_dir is not None:
        os.chdir(work_dir)

    module = importlib.import_module(MODULES[experiment])
    fn = getattr(module, RUNNERS[(experiment, task)])
    call_runner(fn, task, params)


if __name__ == "__main__":
    main()
