# ML-NIDS Temporal Evaluation Pipeline

## Overview

This repository implements a machine learning pipeline for evaluating
Network Intrusion Detection Systems (NIDS) under temporal drift using
the CSE-CICIDS-2018 dataset.

The pipeline performs the following steps:

    1. Exploratory data analysis
    2. Same-day baseline training and evaluation
    3. Temporal drift evaluation using rolling and sliding windows
    4. Metric analysis and result visualization

All results are written to the `results/` directory.

The dataset is not included in this repository due to its large size. It can be downloaded from:

    https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv

## Project Structure

    ML-NIDS/
    │
    ├── data/                    # Dataset files
    ├── models/                  # Saved trained models
    ├── results/                 # Experiment outputs and plots
    ├── scripts/                 # Standalone scripts
    │
    ├── src/                     # Core pipeline code
    │   ├── preprocess.py
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── metrics.py
    │   ├── eda.py
    │   ├── rolling_window_eval.py
    │   ├── sliding_window_eval.py
    │   ├── drift_analysis.py
    │   ├── threshold_analysis.py
    │   ├── feature_importance.py
    │   ├── roc.py
    │
    ├── run_eda.py
    ├── run_rolling_eval.py
    ├── run_sliding_eval.py
    ├── train_pipeline.py
    ├── test_pipeline.py
    └── README.md

## Environment Setup

1. Create a virtual environment:

        python -m venv .venv

2. Activate the environment:

        .venv\Scripts\activate

3. Install required packages:

        pip install -r requirements.txt

## Dataset

Place the CSE-CICIDS-2018 CSV files inside:

    data/CSE-CICIDS-2018/

Example files used by the experiments:

    02-14-2018.csv
    02-15-2018.csv
    02-16-2018.csv
    02-20-2018.csv
    02-21-2018.csv
    02-22-2018.csv
    02-23-2018.csv
    02-28-2018.csv
    03-01-2018.csv
    03-02-2018.csv

## Execution Order

Run the pipeline in the following order.

### 1. Baseline Training

    python train_pipeline.py
    python baseline_same_day.py

Saved models:

    models/rf_model.pkl
    models/xgb_model.pkl    
    models/rf_same_day.pkl
    models/xgb_same_day.pkl

### 2. Exploratory Data Analysis

    python run_eda.py

Generates:

    - class distribution plots
    - feature statistics
    - exploratory visualizations

Output is saved to `results/`.

### 3. Same Day Baseline Evaluation

    python baseline_same_day.eval.py

Computes operational metrics:

    - pAUC at 1 percent FPR
    - Recall at 1 percent FPR

Outputs:

    results/same_day_baseline_metrics.csv

### 4. Rolling Window Temporal Evaluation

    python run_rolling_eval.py

#### Expanding window experiment.

Training data grows over time while testing on the next day.

Example:

    Train: Day1 Day2 Day3
    Test:  Day4

    Train: Day1 Day2 Day3 Day4
    Test:  Day5

Outputs:

    results/rolling_window_results.csv
    results/rolling_window_pauc.png

### 5. Sliding Window Temporal Evaluation

    python run_sliding_eval.py

#### Fixed size training window.

Example with window size 3:

    Train: Day1 Day2 Day3
    Test:  Day4

    Train: Day2 Day3 Day4
    Test:  Day5

Outputs:

    results/sliding_window_results.csv
    results/sliding_window_pauc.png

## Optional Analysis Script

### Attack Distribution

    python scripts/print_attack_distribution.py

Shows attack types present in each day of the dataset.

## Output Directory

All experiment outputs are saved to:

    results/

## Quick Start

Main experiments:

    python train_pipeline.py
    python baseline_same_day.py
    python run_eda.py
    python baseline_same_day.eval.py
    python run_rolling_eval.py
    python run_sliding_eval.py

This produces the baseline metrics and temporal drift evaluation plots.
