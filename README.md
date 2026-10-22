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
- `notebooks/evaluate_test_set.ipynb` — post-training evaluation of an
  existing checkpoint without retraining.

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

Set `use_cross_validation_ensemble = True` in `src/config.py` to evaluate the
held-out test set using the average predictions of all fold checkpoints. The
ensemble is saved as `saved_weights/cross_validation_ensemble.pth`; the
single best-fold checkpoint remains available as
`saved_weights/best_prostate_unet3d.pth`.

If you do not want the final convenience copy named
`best_prostate_unet3d.pth`, set `save_best_checkpoint_alias = False` in
`ExperimentConfig`. Training will still save the fold checkpoint files
(`cross_validation_fold_<n>.pth`) and use the selected best fold for the final
test evaluation.

Adapt the model, data format, and metrics to your specific task.

## Test-set evaluation artifacts

After training, the default configuration evaluates each held-out test volume
individually and writes artifacts under `evaluation/`:

- `test_case_metrics.csv` — Dice, precision, recall, HD95, ASSD, volume error,
  and false-positive/false-negative voxel counts for every test volume.
- `patient_detection_metrics.csv` — patient-level sensitivity, specificity,
  accuracy, and the true/false positive/negative patient counts.
- `test_metric_distributions.png` — box plots of the per-volume metrics.
- `volume_agreement.png` — predicted versus ground-truth lesion volume.
- `*_views.png` — axial, coronal, and sagittal input/ground-truth/prediction/
  overlay figures for representative test volumes.
- `*_surface_error.png` — 3D predicted surfaces colored by distance from the
  ground-truth surface.

Surface distances are reported in millimetres using `target_spacing`. Set
`generate_evaluation_artifacts = False` in `ExperimentConfig` to disable this
step. The evaluator uses the selected best-fold checkpoint and the configured
`evaluation_threshold` (default `0.5`). Select the threshold using validation
data, not the held-out test set.

The same evaluation can be run after training from
`notebooks/evaluate_test_set.ipynb`. Set the checkpoint path, test directory,
and output directory in the first code cell, then execute the notebook cells
in order. This is useful when training has already completed or when comparing
multiple saved checkpoints.

## Interactive prediction viewer

For interactive inspection of every test slice, run the evaluation notebook
through the evaluation cell, then run:

```python
from src.utils.evaluation import show_prediction_viewer

show_prediction_viewer(
    config=config,
    checkpoint=CHECKPOINT_PATH,
    test_files=TEST_FILES,
    device=device,
)
```

The viewer provides case, orientation, background-channel, and slice controls.
The available backgrounds are ADC, DWI, T2, and anatomy. It displays the
selected background, ground truth, prediction, and an overlay. Ground truth is
green, prediction is red, and overlapping regions are yellow. Predictions are
computed once when the viewer is created, so changing slices or backgrounds
does not rerun the model. Enable **Focus on overlay** to hide the other panels
and enlarge the combined overlay for closer inspection.

For a rotatable 3D comparison based on the existing Plotly single-3D display,
run:

```python
from src.utils.evaluation import show_prediction_3d_viewer

show_prediction_3d_viewer(
    config=config,
    checkpoint=CHECKPOINT_PATH,
    test_files=TEST_FILES,
    device=device,
)
```

This displays the ground-truth lesion as a green surface and the predicted
lesion as a red surface. Use the Plotly controls to rotate, zoom, and inspect
the two meshes. Cases can be changed from the dropdown. Use the
**Ground truth** and **Prediction** checkboxes, or click the corresponding
legend entries, to hide or show either layer. Use **Axial scale** to visually
exaggerate the low-resolution axial direction when the lesion is hard to see;
this changes display geometry only, not masks or metrics.
