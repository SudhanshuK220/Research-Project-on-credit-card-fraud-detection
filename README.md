# Credit Card Fraud Detection: A Research Project on Comparative Analysis of ML Models

This repository contains a **research-based project**. It is a reproducible machine learning pipeline that compares five classifiers (SVM, Decision Tree, KNN, Random Forest and XGBoost) for credit card fraud detection on the ULB dataset, using statistical validation, cost-sensitive threshold analysis and SHAP explainability..

All results reported in this README are the **research results of this project**, obtained with the code in this repository on the ULB dataset (80/20 stratified hold-out split and 10-fold stratified cross-validation).

## Research Objectives

The project addresses weaknesses commonly found in fraud detection studies:

- Accuracy-only evaluation despite extreme class imbalance
- Single train-test splits with no statistical validation of model comparisons
- No cost-sensitive evaluation, although a missed fraud is far costlier than a false alarm
- No explanation of model decisions

## Overview

Fraud makes up only **0.172%** of the ULB Credit Card Fraud dataset (492 of 284,807 transactions, a 578:1 imbalance), so accuracy is a misleading metric: a model that labels every transaction as legitimate scores over 99.8% accuracy and catches no fraud. This project evaluates the models with imbalance-aware metrics and checks whether the differences between them are statistically real.

**What the pipeline does:**

- Leak-free, **fold-internal SMOTE** (applied only to training data)
- **10-fold stratified cross-validation** (`shuffle=True`, `random_state=42`)
- **PR-AUC and MCC** alongside Accuracy, Precision, Recall, F1 and ROC-AUC
- **Wilcoxon signed-rank tests** on all 10 model pairs (F1, MCC, PR-AUC)
- **Optimal F1 threshold** selection and **cost-sensitive analysis** (FN cost = 10, FP cost = 1)
- **SHAP explainability** (beeswarm, bar, waterfall and cross-model heatmap)

**Models:** SVM (LinearSVC + calibration), Decision Tree, KNN, Random Forest, XGBoost.

**Preprocessing:** `Time` is converted to an `Hour` feature (hour of day, mod 24); `Amount` and `Hour` are scaled with RobustScaler; V1–V28 are left unscaled. The data is split 80/20 (stratified).

## Key Research Results

### Hold-out test set (20% split, default threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| SVM | 0.9724 | 0.0545 | 0.9184 | 0.1029 | 0.2200 | 0.9741 | 0.7231 |
| Decision Tree | 0.9970 | 0.3432 | 0.8265 | 0.4850 | 0.5315 | 0.9119 | 0.2840 |
| KNN | 0.9983 | 0.5122 | 0.8571 | 0.6412 | 0.6619 | 0.9383 | 0.5899 |
| Random Forest | 0.9993 | 0.7736 | 0.8367 | 0.8039 | 0.8042 | 0.9735 | 0.8598 |
| XGBoost | 0.9989 | 0.6439 | 0.8673 | 0.7391 | 0.7469 | 0.9774 | 0.8630 |

### 10-fold stratified cross-validation (mean ± std)

| Model | F1 | PR-AUC | MCC |
|---|---|---|---|
| SVM | 0.1426 ± 0.0103 | 0.7266 ± 0.0513 | 0.2618 ± 0.0133 |
| Decision Tree | 0.5259 ± 0.0346 | 0.3099 ± 0.0344 | 0.5544 ± 0.0310 |
| KNN | 0.6663 ± 0.0355 | 0.6088 ± 0.0565 | 0.6813 ± 0.0337 |
| Random Forest | **0.8309 ± 0.0395** | 0.8505 ± 0.0416 | **0.8308 ± 0.0396** |
| XGBoost | 0.7648 ± 0.0495 | **0.8513 ± 0.0415** | 0.7684 ± 0.0473 |

### Statistical significance (Wilcoxon signed-rank, n = 10 folds)

- All pairwise comparisons are significant on F1 and MCC (p ≤ 0.003906).
- On PR-AUC, all pairs are significant **except Random Forest vs XGBoost (p = 0.6953)**, indicating comparable discrimination ability for fraud detection.

### Main findings

- **Random Forest** has the best CV F1 and MCC; **XGBoost** has the best PR-AUC and the lowest expected cost.
- **SVM paradox:** ROC-AUC 0.9741 and accuracy 0.9724, yet F1 is only 0.1029 (1,562 false positives at the default threshold). The cause is combining SMOTE with `class_weight='balanced'` in LinearSVC. Use one or the other, not both.
- **V14** is the most important predictor for all tree-based models in the SHAP bar-plot analysis (largest mean |SHAP|).

## Project Structure

```
Research-Project-on-credit-card-fraud-detection/
│
├── creditcard.csv                    ← Kaggle dataset (you must add this)
│
├── credit_card_preprocessing.py      ← Data loading, SMOTE, scaling, evaluate_model()
├── train_SVM.py                      ← Train Support Vector Machine
├── train_KNN.py                      ← Train K-Nearest Neighbors
├── train_DecisionTree.py             ← Train Decision Tree
├── train_ensemble_models.py          ← Train Random Forest + XGBoost
│
├── kfold_cross_validation.py         ← 10-Fold CV + Wilcoxon tests
├── shap_explainability.py            ← SHAP plots for all 5 models
├── evaluation_table.py               ← Comparative table (incl. MCC + PR-AUC)
├── research_visuals.py               ← Metric charts, confusion matrices, threshold, cost,
│                                       cross-validation and Wilcoxon figures
│
├── run_all.py                        ← Master script, runs everything at once
├── requirements.txt                  ← All dependencies
│
├── model_performance_results.json    ← Results from the project's run (pre-generated)
├── kfold_results.json
├── comparative_evaluation_table.csv
├── kfold_summary.csv
│
└── figures/                          ← Pre-generated charts (PNG files)
    └── shap/                         ← SHAP plots and feature importance charts
```

## Output Files Generated After Running

| File | Description |
|---|---|
| `model_performance_results.json` | Raw results for all 5 models |
| `comparative_evaluation_table.csv` | Full metrics table (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC) |
| `kfold_results.json` | Per-fold scores for all models |
| `kfold_summary.csv` | K-Fold Mean ± Std table |
| `shap_beeswarm_*.png` | SHAP beeswarm plots (1 per model) |
| `shap_bar_*.png` | SHAP feature importance bars (1 per model) |
| `shap_waterfall_*.png` | SHAP waterfall, single-prediction explanation |
| `shap_heatmap_all_models.png` | Cross-model top-10 features heatmap |
| `Accuracy_chart.png`, `F1_Score_chart.png`, etc. | Metric bar charts |
| `paper_confusion_matrices.png` | All confusion matrices side by side |

The repository already includes the results of the project's run: the `.json` and `.csv` files in the project folder and the charts in `figures/`. Running the scripts regenerates the `.json` and `.csv` files in place and saves freshly generated charts in the project folder. The `.json` and `.csv` files must stay in the project folder, because `research_visuals.py` reads them from there.

## Step-by-Step Setup in VS Code

### STEP 1: Install Python
Make sure Python 3.9 or higher is installed. Check by opening the VS Code terminal and running:

```
python --version
```

If not installed, download it from https://www.python.org/downloads/

### STEP 2: Open the project folder in VS Code
1. Download the repository (**Code → Download ZIP**) and extract it to a folder, or clone it with Git
2. Open VS Code
3. Click **File → Open Folder**
4. Select the extracted project folder and click **Select Folder**

### STEP 3: Add the dataset
1. Download `creditcard.csv` from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` directly inside the project folder (same level as the `.py` files)

### STEP 4: Open the terminal in VS Code
Press ``Ctrl + ` `` (backtick) or go to **View → Terminal**.

### STEP 5: Create a virtual environment (recommended)

Windows:
```
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

### STEP 6: Install all dependencies
```
pip install -r requirements.txt
```
This may take 3–5 minutes.

### STEP 7: Select the virtual environment in VS Code
1. Press `Ctrl + Shift + P`
2. Type **Python: Select Interpreter**
3. Choose the one that shows `venv` in the path

### STEP 8: Run the full pipeline
```
python run_all.py
```

Expected runtime:

| Step | Approximate time |
|---|---|
| Training 5 models | 5–10 minutes |
| 10-Fold Cross-Validation | 15–25 minutes |
| SHAP analysis | 10–20 minutes (SVM/KNN are slow) |
| Charts and visuals | 1–2 minutes |
| **Total** | **~35–60 minutes** |

`run_all.py` generates the metric bar charts and the confusion matrices. To generate the remaining figures (Wilcoxon p-value heatmap, threshold and cost charts, cross-validation charts), also run `python research_visuals.py` once `run_all.py` has finished.

### STEP 9: Run individual scripts (optional)
```
# Train individual models only
python train_SVM.py
python train_KNN.py
python train_DecisionTree.py
python train_ensemble_models.py     # Random Forest + XGBoost

# Generate evaluation table
python evaluation_table.py

# Run K-Fold CV + statistical tests only
python kfold_cross_validation.py

# Run SHAP analysis only
python shap_explainability.py

# Generate all charts and figures
python research_visuals.py
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'xgboost'`**: run `pip install xgboost`
- **`ModuleNotFoundError: No module named 'shap'`**: run `pip install shap`
- **`creditcard.csv` not found**: make sure the CSV is in the same folder as the `.py` files
- **SHAP is very slow for KNN/SVM**: this is normal, since KernelExplainer is model-agnostic. To speed it up, lower the `sample_size` value in `shap_explainability.py` (for example to 100).
- **VS Code shows red underlines on imports**: make sure you selected the correct Python interpreter (Step 7)

## Core Components

| Component | File | Purpose |
|---|---|---|
| Random Forest and XGBoost | `train_ensemble_models.py` | Ensemble comparison |
| MCC metric | `credit_card_preprocessing.py` | Better metric for imbalanced data |
| PR-AUC metric | `credit_card_preprocessing.py` | More reliable than ROC-AUC for fraud |
| 10-Fold CV | `kfold_cross_validation.py` | Statistical rigour (mean ± std) |
| Wilcoxon test | `kfold_cross_validation.py` | Tests significance of differences |
| SHAP analysis | `shap_explainability.py` | Model explainability |
| Master script | `run_all.py` | Runs the entire pipeline with one command |

## Optimal F1 Threshold and Expected Cost

Practical deployment requires adjusting the classification threshold. For each model, the threshold τ* that maximises F1 is found on the Precision-Recall curve. A simple cost matrix is then applied: a missed fraud (false negative) costs 10 and a false alarm (false positive) costs 1, so

```
Expected Cost = 10 × FN + 1 × FP
```

| Model | τ* | Expected cost |
|---|---|---|
| SVM | 0.9995 | 183 |
| Decision Tree | 1.0000 | 325 |
| KNN | 1.0000 | 180 |
| Random Forest | 0.7459 | 216 |
| **XGBoost** | 0.9801 | **214** |

**Reading the table:**
- XGBoost gives the lowest expected cost (214), with Random Forest not far behind (216), a cost reduction of 34% from the Decision Tree (325).
- For Decision Tree and KNN, τ* = 1.0, which indicates a lack of calibration.
- SVM has a lower cost (183) at τ* = 0.9995, but its practical applicability suffers from high sensitivity to the threshold.
- The cost matrix (C_FN = 10, C_FP = 1) is a simplification.

## Limitations

- Limited interpretability, because the V1–V28 features are PCA-anonymised.
- Limited applicability of the results, because the data is a single European dataset covering 48 hours.
- The cost matrix (C_FN = 10, C_FP = 1) is overly simplistic, and the limited time period does not account for concept drift.

## Future Work

Deep learning models (autoencoders, GNNs), cost-sensitive training, temporal splits for concept drift, and testing on additional fraud datasets.

## Dataset

ULB Machine Learning Group, *Credit Card Fraud Detection*, Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
