# AI Project

A starter project for data analysis and training AI models in Python. Datasets and saved model weights are not stored in the repository.

## Local Setup

Python 3.10 or newer is required. Run these commands from the project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python main.py
```

For a PyTorch installation tailored to your GPU, use the installer selector on the PyTorch website. The version installed from `requirements.txt` may run on CPU only, depending on your platform.

## Data

- `data/raw/` — original source data; never modify these files.
- `data/processed/` — preprocessed data ready for training.
- `models/saved_weights/` — model checkpoints and weights.

The preprocessing pipeline creates a reproducible, patient-level held-out test
split (seed `42`) before preprocessing. Its outputs are stored under
`data/processed/train_val/` and `data/processed/test/`. Training creates an
ordinary train/validation split or cross-validation folds from `train_val/`.
Adjust `validation_fraction`, `test_fraction`, and `split_seed` in
`My_preprocessing/settings/main_settings.py` to change this behavior.

The contents of these directories are ignored by Git. Do not store patient data or other sensitive information in a public repository.

## Google Colab

The Colab workflow is split into two notebooks:

1. Open `notebooks/colab/01_preprocessing.ipynb`.
2. Store `MGR_dataset_correct.zip` at
   `MyDrive/Master_Thesis/Datasets/`.
3. Run the preprocessing notebook. It keeps source code and raw data in
   Colab's local storage, then writes `processed_train_val.tar.gz` and
   `processed_test.tar.gz` to
   `MyDrive/Master_Thesis/processed_archives/`.
4. Open `notebooks/colab/02_train.ipynb`. It extracts those archives into
   local Colab storage and saves checkpoints to
   `MyDrive/Master_Thesis/training_checkpoints/`.

The processed archives are kept separately so the training notebook can
restore the train, validation, and test splits without rerunning preprocessing.

Both notebooks include an explicit **Optional: update the local checkout**
cell. Run it after pushing code changes to GitHub and before importing project
modules. It uses `git pull --ff-only`, so it will stop instead of overwriting
local changes. Set `UPDATE_REPOSITORY = False` when you want to keep the
current checkout.

## Code Layout

- `src/config.py` — project location and data/model paths.
- `src/data/` — data loading and preprocessing.
- `src/models/` — model architectures and training loop.
- `src/utils/` — metrics and shared utilities.
- `notebooks/` — data exploration and training entry point.

## Training progress and logs

Run training with `python main.py`. Each epoch displays separate progress bars
for training and validation batches. Timestamped epoch summaries and checkpoint
events are written both to the console and to `logs/training.log`.

Training runs five-fold patient-level cross-validation by default. The `train`
and `validation` samples form the cross-validation pool; the `test` split is
held out and evaluated only after all folds finish. Fold checkpoints are saved
under `saved_weights/cross_validation_fold_<n>.pth`, and the best fold is copied
to `saved_weights/best_prostate_unet3d.pth`. Configure
`cross_validation_folds` and `cross_validation_seed` in
`src/config.py` to change this behavior.

Adapt the model, data format, and metrics to your specific task.
