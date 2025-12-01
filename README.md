# CMPT 459 – Obesity Analysis (Group 8)

Lightweight README (not a report). Covers setup, environment, how to run, repo map, and data notes.

## Requirements
- Python 3.9+ recommended
- Packages: pandas, numpy, matplotlib, seaborn, scikit-learn
- Jupyter (only if you want to run `EDA.ipynb`)

## Setup
```bash
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data
- `cleaned_data.csv` (primary input for scripts)
- `ObesityDataSet_raw_and_data_sinthetic.csv` (raw/original)
- Files are already in the repo; no extra download needed.

## How to run
### Full pipeline
Runs clustering, outlier detection, feature selection, classification (k-NN, Logistic Regression, Random Forest), and hyperparameter tuning.
```bash
python main.py
```
### Notebook (EDA)
```bash
jupyter notebook EDA.ipynb
```

## Repository map
- `main.py` – primary pipeline orchestrator.
- `classification/` – model code: k-NN, logistic regression, random forest.
- `clustering/` – hierarchical, DBSCAN, CLARANS clustering.
- `feature_selection/` – mutual information, LASSO feature selection.
- `outlier_detection/` – k-NN distance, probabilistic (GMM), LOF outlier detection.
- `hyperparameter_tuning/` – tuning modules (k-NN, random forest).
- `cleaned_data.csv` – processed dataset used by pipelines.
- `ObesityDataSet_raw_and_data_sinthetic.csv` – original/raw dataset.
- `EDA.ipynb` – exploratory data analysis notebook.

## Environment notes
- All scripts expect to be run from the repo root so relative paths to `cleaned_data.csv` resolve.