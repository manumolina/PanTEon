# Running PanTEon with Docker

This guide explains how to build and run the PanTEon CPU container.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- Enough disk space for the image (~4–6 GB compressed; first build downloads TensorFlow, PyTorch, and Transformers)
- Optional external files (not bundled in the image):
  - **PanTEon Database FASTA** — download from [Zenodo](https://zenodo.org/records/18039747) and mount into `data/`
  - **Pre-trained models** — download from Zenodo and mount at runtime
  - **`kanalyze.jar`** — required for ClassifyTE; place under `features/kanalyze-2.0.0/code/`
  - **`ltrsearch` and `itrsearch`** — required for NeuralTE training; place under `tools/` and make executable

## Build the image

From the repository root:

```bash
docker build -t panteon:cpu .
```

The build uses a multi-stage `Dockerfile` with `python:3.10-slim-bookworm` and installs CPU-only PyTorch and TensorFlow.

## Basic usage

The container entrypoint is `python PanTEon.py`. Pass any PanTEon subcommand and its arguments after the image name:

```bash
docker run --rm panteon:cpu <module> [options]
```

Show help for a module:

```bash
docker run --rm panteon:cpu training -h
docker run --rm panteon:cpu inference -h
docker run --rm panteon:cpu library -h
docker run --rm panteon:cpu evaluation -h
```

## Recommended volume mounts

| Host path | Container path | Purpose |
|-----------|----------------|---------|
| `/path/to/output` | `/work` | Working directory and results |
| `/path/to/models` | `/models` | Pre-trained or output models |
| `/path/to/database` | `/opt/PanTEon/data` | PanTEon Database FASTA and metadata |

Example with volumes:

```bash
docker run --rm \
  -v "$(pwd)/work:/work" \
  -v "$(pwd)/models:/models" \
  -v "$(pwd)/data:/opt/PanTEon/data" \
  panteon:cpu inference \
    -f /work/input.fasta \
    -t 4 \
    -w /work \
    -d /models \
    -n all \
    -p results
```

## Examples

### Quick test (bundled toy data)

Training:

```bash
docker run --rm \
  -v "$(pwd)/Test_data:/opt/PanTEon/Test_data" \
  panteon:cpu training \
    -f Test_data/sequences_toy.fasta \
    -t 2 \
    -d Test_data/testing_models \
    -w Test_data/work_dir \
    -n all
```

Inference:

```bash
docker run --rm \
  -v "$(pwd)/Test_data:/opt/PanTEon/Test_data" \
  panteon:cpu inference \
    -f Test_data/sequences_toy.fasta \
    -t 2 \
    -d Test_data/testing_models \
    -w Test_data/work_dir \
    -n all \
    -p testing
```

### Library module

Mount the full database directory if the large FASTA is stored on the host:

```bash
docker run --rm \
  -v "/path/to/PanTEon/data:/opt/PanTEon/data" \
  panteon:cpu library \
    --taxon Plantae \
    --req_class LTR \
    --view_only
```

### Evaluation module

```bash
docker run --rm \
  -v "$(pwd)/work:/work" \
  panteon:cpu evaluation \
    --true_fasta /work/true.fasta \
    --pred_fasta /work/pred.fasta
```

## Interactive shell

To inspect the container or run commands manually:

```bash
docker run --rm -it \
  -v "$(pwd)/work:/work" \
  --entrypoint /bin/bash \
  panteon:cpu
```

Inside the shell, PanTEon is available as:

```bash
python PanTEon.py inference -h
```

## Notes

- **CPU only**: this image is optimized for size and runs on CPU. Training large models will be slow; GPU support is not included.
- **Thread count**: set `-t` to the number of CPU cores you want PanTEon to use.
- **Persist outputs**: always mount a host directory to `-w` / `--work-dir` so results are kept after the container exits.
- **Custom classifiers**: mount `Custom_classifiers/` if you use your own models:

  ```bash
  -v "$(pwd)/Custom_classifiers:/opt/PanTEon/Custom_classifiers"
  ```
