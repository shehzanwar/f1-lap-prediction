# F1 Lap Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Academic%20Project-lightgrey?style=flat-square)

A machine learning project that predicts Formula 1 race outcomes — fastest lap time, finishing position, and podium probability — using historical lap timing, qualifying, and driver data from the Ergast F1 dataset.

> Developed as a course project for Georgia Tech CSE/ISYE 7406: Data Mining and Statistical Learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Dataset](#dataset)
7. [Quick Start](#quick-start)
8. [Usage](#usage)
9. [Modeling Approach](#modeling-approach)
10. [Results](#results)
11. [Troubleshooting](#troubleshooting)
12. [Roadmap](#roadmap)
13. [Contributing](#contributing)
14. [License](#license)
15. [Acknowledgments](#acknowledgments)

---

## Overview

This project applies supervised machine learning techniques to historical Formula 1 data to address three related prediction problems:

- **Regression (Q1):** Predict a driver's fastest lap time in milliseconds for a given race.
- **Regression (Q2):** Predict a driver's final finishing position.
- **Binary Classification (Q3):** Predict whether a driver will finish on the podium (top 3).

The pipeline covers the full data science workflow: raw CSV ingestion, data cleaning, feature engineering, exploratory analysis, and pre-modeling data preparation with imputation, train/test splitting, and feature scaling.

---

## Features

- Ingests and merges multiple Ergast F1 CSV datasets (results, qualifying, races, drivers, constructors, lap times) into a single analysis-ready master DataFrame.
- Converts race time strings in `M:SS.mmm` and `H:MM:SS.mmm` formats to milliseconds for consistent numeric handling.
- Engineers aggregate lap-timing features (per-race mean and standard deviation of lap times per driver).
- Computes driver age at race date as a continuous predictor.
- Generates a correlation heatmap across all key numeric variables and saves it as a PNG artifact.
- Applies median imputation to handle missing values and standardizes all features via `StandardScaler` prior to modeling.
- Supports three separate target variables from a single cleaned dataset without code duplication.

---

## Repository Structure

```
f1-lap-prediction/
├── f1datasets/                          # Directory of raw Ergast CSV files
│   ├── results.csv
│   ├── qualifying.csv
│   ├── races.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   ├── lap_times.csv
│   └── ...
├── f1datasets.zip                       # Compressed archive of the above
├── f1_script.py                         # Standalone Python pipeline script
├── f1_proj.ipynb                        # Full analysis notebook with outputs
├── f1_correlation_heatmap.png           # Generated correlation heatmap
├── 7406 Project Proposal-2025F.pdf      # Project proposal document
├── 7406 Project Written Report-2025F.pdf # Final written report
└── Anwar_Shehzad_MidtermReport.pdf      # Midterm progress report
```

---

## Prerequisites

- Python 3.9 or higher
- `pip` (Python package manager)
- Jupyter Notebook or JupyterLab (to run `f1_proj.ipynb`)

The following Python libraries are required:

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, and merging |
| `numpy` | Numerical operations and NaN handling |
| `scikit-learn` | Imputation, scaling, and train/test splitting |
| `seaborn` | Correlation heatmap visualization |
| `matplotlib` | Plot rendering and file output |

---

## Installation

**1. Clone the repository:**

```bash
git clone https://github.com/shehzanwar/f1-lap-prediction.git
cd f1-lap-prediction
```

**2. (Recommended) Create and activate a virtual environment:**

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install required dependencies:**

```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

**4. Extract the dataset (if not already present):**

```bash
unzip f1datasets.zip
```

Confirm that a `f1datasets/` directory now exists in the project root containing the CSV files before proceeding.

---

## Dataset

This project uses the **[Ergast Motor Racing Developer API](http://ergast.com/mrd/)** dataset, a comprehensive historical database of Formula 1 race results dating back to 1950.

The following CSV files are used and must be present in the `f1datasets/` directory:

| File | Description |
|---|---|
| `results.csv` | Race results including finishing position, fastest lap time, and grid position |
| `qualifying.csv` | Q1, Q2, and Q3 qualifying session times per driver per race |
| `races.csv` | Race metadata including circuit, date, year, and round number |
| `drivers.csv` | Driver biographical data including nationality and date of birth |
| `constructors.csv` | Constructor (team) names and nationalities |
| `lap_times.csv` | Individual lap times for every driver in every race |

The compressed archive `f1datasets.zip` in this repository contains all required files.

---

## Quick Start

To run the full data preparation and visualization pipeline in one command:

```bash
python f1_script.py
```

This will:

1. Load all CSV files from `f1datasets/`.
2. Clean and standardize time formats.
3. Engineer lap-timing and driver age features.
4. Merge all datasets into a single master DataFrame.
5. Print a data sample and descriptive statistics to the console.
6. Save `f1_correlation_heatmap.png` to the project root.
7. Perform median imputation, train/test split, and feature scaling, printing confirmation of shapes and sample data.

To run the full analysis interactively (including model outputs):

```bash
jupyter notebook f1_proj.ipynb
```

---

## Usage

### Running the Preprocessing Script

```bash
python f1_script.py
```

**Expected console output (abbreviated):**

```
Loading 14 CSV files...
Cleaning data...
Converting time strings to milliseconds...
Engineering features from 'lap_times'...
Merging DataFrames...
Data preparation complete. Master DataFrame ready for analysis.
Creating final DataFrame for modeling...
Final DataFrame shape: XXXXX rows and 13 columns.
Generating output visualizations...
'f1_correlation_heatmap.png' is saved.
Preparing data for modeling...
X shape: (XXXXX, 9), y1 shape: (XXXXX,)
Missing data imputed using median.
Data split: XXXXX training samples and XXXXX testing samples.
Feature scaling complete. Data is ready for modeling.
```

### Prediction Targets

After running the script, three target arrays are available for downstream modeling:

| Variable | Type | Description |
|---|---|---|
| `y1` (`fastestLapTime_ms`) | Continuous | Fastest lap time in milliseconds |
| `y2` (`positionOrder`) | Ordinal/Continuous | Final finishing position |
| `y3` (`podium`) | Binary (0/1) | 1 if the driver finished in the top 3, 0 otherwise |

### Predictor Features

The following features are included in the model input matrix `X`:

| Feature | Description |
|---|---|
| `year` | Season year |
| `circuitId` | Numeric circuit identifier |
| `grid` | Starting grid position |
| `qualifying_position` | Final qualifying classification |
| `q1_ms` | Q1 session time in milliseconds |
| `q2_ms` | Q2 session time in milliseconds |
| `q3_ms` | Q3 session time in milliseconds |
| `avg_lap_time_ms` | Mean race lap time for the driver (engineered) |
| `std_lap_time_ms` | Standard deviation of race lap times (engineered) |
| `driver_age_at_race` | Driver age in years at race date (engineered) |

---

## Modeling Approach

The preprocessing pipeline in `f1_script.py` prepares data for three distinct supervised learning tasks. The full modeling experiments, hyperparameter tuning, and evaluation are conducted in `f1_proj.ipynb`.

**Pipeline summary:**

1. **Data Loading** — All CSVs in `f1datasets/` are read into a dictionary of DataFrames using `glob`.
2. **Cleaning** — The `\N` placeholder (Ergast null marker) is replaced with `NaN`. Time strings are converted to milliseconds via a custom parser supporting both `M:SS.mmm` and `H:MM:SS.mmm` formats.
3. **Feature Engineering** — Per-driver, per-race lap time statistics (mean and standard deviation) are computed from `lap_times.csv`. Driver age at race date is derived from `drivers.dob` and `races.date`.
4. **Merging** — Results, qualifying, race metadata, driver, constructor, and lap statistics DataFrames are merged on shared keys (`raceId`, `driverId`, `constructorId`).
5. **Imputation** — Missing values in predictor features are filled using median imputation (`SimpleImputer`, strategy = `'median'`).
6. **Splitting** — Data is split 70% training / 30% testing with `random_state=7406`.
7. **Scaling** — Features are standardized using `StandardScaler` fit on the training set only.

---

## Results

[Add screenshot here: f1_correlation_heatmap.png]

The correlation heatmap (`f1_correlation_heatmap.png`) illustrates relationships between all numeric features and target variables. Qualifying session times and grid position exhibit strong positive correlation with finishing position and fastest lap time.

For full model evaluation metrics (RMSE, R-squared, accuracy, AUC), refer to the written reports included in the repository:

- `7406 Project Written Report-2025F.pdf` — Final report with complete methodology and results.
- `Anwar_Shehzad_MidtermReport.pdf` — Midterm progress report.

---

## Troubleshooting

**`FileNotFoundError` when loading CSVs**

Ensure the `f1datasets/` directory exists in the project root and contains the required CSV files. If you downloaded the repository as a ZIP, extract `f1datasets.zip` manually:

```bash
unzip f1datasets.zip
```

**`ModuleNotFoundError` for any dependency**

Confirm your virtual environment is active and install missing packages individually:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

**`KeyError: 'results'`**

The script reads filenames as dictionary keys (e.g., `results.csv` becomes `'results'`). If the `results.csv` file is renamed or missing, this key will not exist. Verify that all required CSVs listed in the [Dataset](#dataset) section are present.

**Large notebook (`f1_proj.ipynb`) fails to render on GitHub**

The notebook is approximately 19 MB due to stored outputs. Render it locally using `nbconvert`, or clear outputs before pushing:

```bash
jupyter nbconvert --to html f1_proj.ipynb
```

---

## Roadmap

- [ ] Add explicit model training cells to `f1_script.py` for all three targets (linear regression, random forest, logistic regression).
- [ ] Include cross-validation and hyperparameter tuning.
- [ ] Add a `requirements.txt` for reproducible environment setup.
- [ ] Export trained model artifacts via `joblib` for inference without re-running the full pipeline.
- [ ] Add unit tests for the `time_to_ms` parsing function.
- [ ] Extend the dataset to include pit stop data and safety car laps as additional features.

---

## Contributing

This repository was created for an academic course project and is not currently accepting external contributions. If you find a bug or have a suggestion, feel free to open an issue.

If you wish to build on this work, please fork the repository and adapt it to your own needs.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **[Ergast Motor Racing Developer API](http://ergast.com/mrd/)** for providing the comprehensive historical F1 dataset used in this analysis.
- **Georgia Tech CSE/ISYE 7406: Data Mining and Statistical Learning** course staff for the project framework and guidance.
- The `scikit-learn`, `pandas`, and `seaborn` open source communities for the libraries that power this pipeline.
