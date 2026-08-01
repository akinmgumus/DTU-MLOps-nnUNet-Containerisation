# Containerised nnU-Net Training & Inference

GPU Docker images that take an nnU-Net drone-imagery segmentation model from raw Kaggle data to trained weights, and from raw RGB photographs to colour-overlaid segmentation masks — each as a single `docker run`.

**Scope and attribution.** This is **my slice of a five-person group project** for the MLOps course at DTU (January 2026), not the whole system. The group's full pipeline also covered DVC data/model versioning, GitHub Actions CI, a BentoML serving layer, a FastAPI service, GCP deployment and data-drift monitoring — **none of that is mine and none of it is here.**

What is mine, and what this repository contains:

| | |
|---|---|
| `train.dockerfile`, `train_entrypoint.sh` | The training image and its three-stage entrypoint — written by me |
| `inference.dockerfile`, `inference_entrypoint.sh` | The inference image and its pipeline — written by me |
| `src/dtu_mlops_111/trainers.py` | Custom nnU-Net trainer subclass with W&B + loguru logging — mostly mine |
| `src/dtu_mlops_111/run_inference.py`, `predictors.py` | Inference entry point and predictor — partly mine |
| `prepare_inference_input.py`, `visualize_results.py` | RGB→nnU-Net channel conversion; mask overlay rendering — mostly mine |
| `docs/DOCKER_TRAINING.md`, `docs/DOCKER_INFERENCE.md` | Operating instructions — written by me |

The original group repository is [DTU_MLOps_111](https://github.com/ecenazelverdi/DTU_MLOps_111); its commit history shows the split. This repository does not carry that history over, because doing so would republish four other people's commits under my name.

## Why containerising nnU-Net is not trivial

nnU-Net is opinionated: it discovers datasets, plans, and trainer classes by convention, through environment variables and its own package layout. Three things had to be solved.

**1. Injecting a custom trainer.** nnU-Net finds trainer classes by name inside its own installed package. There is no plugin hook. Rather than fork the library, the build copies my trainer into nnU-Net's package tree at image-build time:

```dockerfile
RUN cp /app/src/dtu_mlops_111/trainers.py \
    $(python3 -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")/training/nnUNetTrainer/variants/custom_trainer.py
```

`nnUNetv2_train ... -tr nnUNetTrainer_5epochs_custom` then resolves it like any built-in trainer. The subclass caps epochs at 5 (the library default is 1000), auto-selects CUDA → MPS → CPU, and overrides `on_epoch_end` to push loss to Weights & Biases and to a rotating loguru file at the same time.

**2. Writable cache paths.** Under a container's default user, matplotlib and torch.inductor try to write caches into a `HOME` that may not be writable, and the run dies partway through. `HOME`, `MPLCONFIGDIR` and `TORCHINDUCTOR_CACHE_DIR` are all redirected to `/tmp`, and the entrypoint creates them before anything else runs.

**3. Not letting host paths leak in.** `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` are pinned to container paths inside the entrypoint and are deliberately *not* read from the passed-in environment — otherwise a host `.env` that happens to define them would silently point the container at directories that do not exist inside it.

## Secrets

Secrets are never baked into an image. `KAGGLE_*` and `WANDB_*` are supplied at run time:

```bash
docker run --gpus all --ipc=host --env-file .env ... droneseg-training
```

This is a deliberate correction to the version submitted for the course, which had `COPY .env /app/.env` in both Dockerfiles. That is unsafe: each `COPY` becomes a permanent image layer, so anyone holding the image can recover the file — deleting it in a later step does not remove the layer. The run commands in `docs/` already passed `--env-file`, so the `COPY` was doing nothing except leaving credentials inside the image. The entrypoints now read the variables straight from the environment and fail fast with a clear message if the Kaggle credentials are missing.

## Training pipeline

```
docker run --env-file .env -v ./nnUNet_raw:/app/nnUNet_raw ...
  │
  ├─ 1. dtu_mlops_111.data main        download from Kaggle, export to nnU-Net layout
  ├─ 2. nnUNetv2_plan_and_preprocess   -d 101 --verify_dataset_integrity
  └─ 3. nnUNetv2_train 101 2d 0        -tr nnUNetTrainer_5epochs_custom --npz
```

Base image is `nvidia/cuda:12.1.0-base-ubuntu22.04`; `uv` is pulled from its official image for dependency installation, and dependencies are installed *before* the source is copied so that source edits do not invalidate the dependency layer. A commented-out loop in the entrypoint trains all five folds for an ensemble.

![W&B training loss](docs/wandb_loss.png)

*Training loss logged to Weights & Biases from the custom trainer's `on_epoch_end`.*

## Inference pipeline

```
docker run --gpus all -v ./nnUNet_results:/nnUnet_results -v ./images_raw:/images_raw ...
  │
  ├─ 1. prepare_inference_input.py     RGB photo → three single-channel nnU-Net inputs
  ├─ 2. dtu_mlops_111.run_inference    custom predictor, checkpoint_best.pth, TTA disabled
  └─ 3. visualize_results.py           segmentation mask → colour overlay on the original
```

The entrypoint checks that a trained model and input images are actually mounted and exits with an actionable message rather than a stack trace if either is missing.

![Segmentation results](docs/wandb_visualization_results.png)

*Segmentation overlays produced by the inference container.*

## Running it

Full instructions, including volume mounts and troubleshooting, are in:

- [`docs/DOCKER_TRAINING.md`](docs/DOCKER_TRAINING.md)
- [`docs/DOCKER_INFERENCE.md`](docs/DOCKER_INFERENCE.md)

Note that this repository is the containerisation layer only. `dtu_mlops_111.data`, invoked in step 1 of training, lives in the group repository and is not reproduced here — so the training image will not build into a runnable pipeline from this repository alone. It is published to show the container and trainer design, not as a standalone product.

## License

MIT — see [LICENSE](LICENSE).
