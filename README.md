# Heart Disease Prediction — KNN Classifier

A machine learning project that predicts whether a patient has heart disease based on clinical measurements. Built using **K-Nearest Neighbors (KNN)** with **Stratified K-Fold Cross Validation** and benchmarked against **Logistic Regression** and **Random Forest** classifiers.

> **Primary evaluation metric: Recall (Sensitivity)** — because missing a real heart disease case (false negative) carries far greater consequences than a false alarm.

---

## Final Model Results

| Model | Accuracy | Precision | Recall | F1 Score | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|
| **KNN** | 87.5% | 88.3% | 89.2% | 88.8% | 85.4% | 0.930 |
| **Logistic Regression** | 89.1% | 88.7% | **92.2%** | 90.4% | 85.4% | **0.933** |
| **Random Forest** | 87.5% | 89.1% | 88.2% | 88.7% | 86.6% | 0.933 |

 **Deployed Model: Logistic Regression** — highest Recall (92.2%) and ROC-AUC (0.933), meaning it correctly identifies the most heart disease patients while staying competitive on all other metrics.

---

## Project Structure

```
heart_disease_knn/
│
├── data/
│   ├── heart.csv                        # Raw dataset (UCI / Kaggle)
│   └── processed/
│       └── processed_data.pkl           # Preprocessed train/test splits (saved)
│
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb           # Cleaning, encoding, scaling
│   ├── 03_knn_model.ipynb               # KNN training + K tuning (K=1 to 30)
│   ├── 04_kfold_evaluation.ipynb        # 10-Fold Stratified Cross Validation
│   └── 05_model_comparison.ipynb        # KNN vs Logistic Regression vs Random Forest
│
├── src/
│   ├── preprocess.py                    # Reusable preprocessing functions
│   ├── train.py                         # Model training functions
│   └── evaluate.py                      # Metrics + confusion matrix + ROC helpers
│
├── models/
│   ├── knn_model.pkl                    # Saved KNN model
│   ├── logreg_model.pkl                 # Saved Logistic Regression model
│   └── rf_model.pkl                     # Saved Random Forest model
│
├── outputs/
│   ├── knn_accuracy_curve.png           # K vs Accuracy / Precision / Recall plots
│   ├── confusion_matrices.png           # Side-by-side confusion matrices (all 3 models)
│   └── model_comparison_table.csv       # Full metrics comparison table
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why it was used |
|---|---|---|
| **Language** | Python 3 | Best ecosystem for ML + data analysis |
| **ML Models** | scikit-learn | KNN, Logistic Regression, Random Forest — all in one library with consistent API |
| **Data Handling** | pandas, numpy | Data loading, cleaning, encoding, and numerical operations |
| **Visualization** | matplotlib, seaborn | Correlation heatmaps, K-curve plots, confusion matrices, ROC curves, feature importance |
| **Scaling** | `StandardScaler` | KNN is distance-based — features must be on the same scale, otherwise large-range features dominate |
| **Encoding** | `pd.get_dummies()` | Converts categorical text columns (Sex, ChestPainType, etc.) into numbers the model can read |
| **Validation** | `StratifiedKFold` | 10-fold cross validation; stratified ensures each fold has the same class ratio as the original data |
| **Model Storage** | `pickle` | Saves trained models to `.pkl` files so they can be reloaded without retraining |
| **Notebooks** | Jupyter | Step-by-step analysis — each notebook covers exactly one stage of the pipeline |

---

## How It Works — Pipeline Overview

```
heart.csv
    │
    ▼
01_eda.ipynb
    Explore distributions, class balance,
    correlations, outliers (Cholesterol=0)
    │
    ▼
02_preprocessing.ipynb
    - Drop duplicates
    - Fix Cholesterol=0 → replace with median
    - One-hot encode: Sex, ChestPainType,
      RestingECG, ExerciseAngina, ST_Slope
    - StandardScale numeric features
    - 80/20 train-test split (stratified)
    - Save → processed_data.pkl
    │
    ▼
03_knn_model.ipynb
    - Baseline KNN (K=5)
    - Loop K=1 to 30 → record Accuracy,
      Precision, Recall per K
    - Plot K vs metrics curves
    - Best K chosen by highest Recall
    - Final KNN saved → knn_model.pkl
    │
    ▼
04_kfold_evaluation.ipynb
    - StratifiedKFold (10 splits)
    - Scaler fitted per fold (no data leakage)
    - Mean ± Std of all metrics across 10 folds
    - sklearn Pipeline for clean cross_val_score
    │
    ▼
05_model_comparison.ipynb
    - KNN vs Logistic Regression vs Random Forest
    - ROC curves (all 3 on same graph)
    - Confusion matrices (side-by-side)
    - Random Forest feature importances
    - Final deployment recommendation
```

---

## Dataset

**Source:** [Heart Failure Prediction Dataset — Kaggle (UCI ML Repository)](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

**918 patients, 11 clinical features, 1 binary target**

| Column | Type | Description |
|---|---|---|
| Age | int | Patient age in years |
| Sex | categorical | M / F |
| ChestPainType | categorical | ATA, NAP, ASY, TA |
| RestingBP | int | Resting blood pressure (mm Hg) |
| Cholesterol | int | Serum cholesterol (mg/dl) — 0 values treated as missing |
| FastingBS | int | Fasting blood sugar > 120 mg/dl (1=True) |
| RestingECG | categorical | Normal, ST, LVH |
| MaxHR | int | Maximum heart rate achieved |
| ExerciseAngina | categorical | Y / N |
| Oldpeak | float | ST depression induced by exercise |
| ST_Slope | categorical | Up, Flat, Down |
| **HeartDisease** | **int** | **Target: 0 = No Heart Disease, 1 = Heart Disease** |

---

## Key ML Concepts Applied

| Concept | What was done |
|---|---|
| **KNN (K-Nearest Neighbors)** | Classifies a patient based on the K most similar patients in the training data using Euclidean distance |
| **Hyperparameter Tuning** | Tested K from 1 to 30; selected best K based on highest Recall, not highest accuracy |
| **Stratified K-Fold (10 folds)** | Splits data 10 different ways to evaluate model reliability; stratification preserves class balance in every fold |
| **Data Leakage Prevention** | `StandardScaler` is fit only on the training fold inside each K-Fold loop — never on test data |
| **Recall / Sensitivity** | Primary metric — of all patients who actually have heart disease, how many did the model correctly identify? |
| **Specificity** | Of all healthy patients, how many did the model correctly say are healthy? |
| **False Negative** | Model predicts "No Heart Disease" when patient actually has it — the most dangerous error in this domain |
| **ROC-AUC** | Overall model quality score; 1.0 = perfect, 0.5 = random guess |
| **Feature Importance** | Random Forest reveals which clinical measurements contribute most to the prediction |

---

## Getting Started

### Prerequisites
- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction/heart_disease_knn

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and place it at:

```
heart_disease_knn/data/heart.csv
```

### Run the Notebooks (in order)

```bash
jupyter notebook
```

Open notebooks in this order:
1. `01_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_knn_model.ipynb`
4. `04_kfold_evaluation.ipynb`
5. `05_model_comparison.ipynb`

---

## 📸 Outputs

| Output | Description |
|---|---|
| `outputs/knn_accuracy_curve.png` | Line plots — K vs Accuracy, K vs Precision, K vs Recall |
| `outputs/confusion_matrices.png` | Side-by-side confusion matrices for all 3 models |
| `outputs/model_comparison_table.csv` | Full metrics table for all models |

---

## 📄 License

This project is open source. Add your preferred license here.

---

## Acknowledgements

- Dataset: [fedesoriano on Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (originally from UCI ML Repository)
- Built as a portfolio project demonstrating healthcare ML evaluation best practices — specifically, why **recall beats accuracy** when false negatives carry real-world consequences.
