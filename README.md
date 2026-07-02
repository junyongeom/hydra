# HYDRA: Hybrid Domain-Robust Activity Recognition via Multimodal RF Sensing under Missing Modalities

The paper is currently under review at IEEE Internet of Things Journal.


Thanks for the wonderful XRF55 dataset: https://github.com/aiotgroup/XRF55-repo

## Layout

```text
configs/                         Experiment configuration files
scripts/run_experiment.py         Config-driven runner
src/hydra/experiments/            Exported and cleaned experiment code
```

The experiment modules were converted from the working notebooks:

- `proposed_HYDRA.ipynb`
- `xrf55_baseline.ipynb`
- `zhang_baseline.ipynb`

Running those notebooks cell-by-cell is sufficient for reproduction; the
Python modules here are provided for easier command-line execution and cleaner
repository structure.

## Data Layout

Set `XRF55_ROOT` or pass `--data-root` to the runner. The expected directory
layout is:

```text
XRF55/
  Scene1/
    WiFi/*.npy
    mmWave/*.npy
    RFID/*.npy
  Scene2/
  Scene3/
  Scene4/
```

The default `data_root` in the config files is `data/XRF55`, relative to the
directory where you run the command. Override it with `--data-root` for local
paths.

## Setup

```bash
pip install -e .
```

If the proposed HYDRA model is used, install the Mamba dependencies listed in
`requirements.txt` in an environment compatible with your CUDA and PyTorch
versions. The proposed HYDRA model requires CUDA because `mamba-ssm` and
`causal-conv1d` use CUDA kernels in this implementation; the baseline models
can run on CPU.

## Running Experiments

Run the proposed HYDRA Cross-Scene & Cross-Subject (CSCS) experiment:

```bash
python scripts/run_experiment.py \
  --config configs/proposed_hydra.yaml \
  --task cscs \
  --data-root /path/to/DATASET/XRF55
```

Run a baseline:

```bash
python scripts/run_experiment.py \
  --config configs/xrf55_baseline.yaml \
  --task cross_scene \
  --data-root /path/to/DATASET/XRF55
```

Available tasks:

- `in_domain`
- `cross_subject`
- `cross_scene`
- `cscs`
- `ablation`
- `sensor_failure`
- `complexity`

Task notes:

- `cscs` runs the cross-scene and cross-subject setup. This is the source of
  the 5-shot fine-tuned checkpoints used by the missing-modality experiment.
- `ablation` is for the proposed HYDRA variants and is configured as CSCS
  5-shot by default.
- `sensor_failure` evaluates CSCS 5-shot checkpoints under missing-modality
  conditions. It reuses the trained `Scene2`, `Scene3`, and `Scene4` 5-shot
  fine-tuned models, then reports performance for WiFi-only, mmWave-only,
  RFID-only, one-sensor-failed pairs, and all-sensors-available inference.
- `complexity` reports parameter count, memory estimate, and GFLOPs for the
  selected model family.

Run sensor-failure evaluation after the corresponding CSCS 5-shot checkpoints
exist:

```bash
python scripts/run_experiment.py \
  --config configs/proposed_hydra.yaml \
  --task sensor_failure \
  --data-root /path/to/DATASET/XRF55 \
  --checkpoint-dir /path/to/reproc \
  --tags base
```

Evaluate saved Cross-Scene checkpoints without copying them into this
repository:

```bash
python scripts/run_experiment.py \
  --config configs/xrf55_baseline.yaml \
  --task cross_scene \
  --data-root /path/to/DATASET/XRF55 \
  --checkpoint-dir /path/to/reproc \
  --shots 5
```
