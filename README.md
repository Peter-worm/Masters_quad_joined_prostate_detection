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

The contents of these directories are ignored by Git. Do not store patient data or other sensitive information in a public repository.

## Google Colab

1. Place the project directory on Google Drive at `MyDrive/ai_project`.
2. Open `notebooks/02_colab_train.ipynb` in Colab.
3. Run the cell that mounts Drive and verify `PROJECT_DIR`.
4. Install dependencies from `requirements.txt` and select a GPU runtime.

Set `AI_PROJECT_DIR` to specify a different project directory. `src/config.py` detects Colab and defaults to `/content/drive/MyDrive/ai_project`.

## Code Layout

- `src/config.py` — project location and data/model paths.
- `src/data/` — data loading and preprocessing.
- `src/models/` — model architectures and training loop.
- `src/utils/` — metrics and shared utilities.
- `notebooks/` — data exploration and training entry point.

Adapt the model, data format, and metrics to your specific task.
