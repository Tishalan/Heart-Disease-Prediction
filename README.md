# ❤️ Heart Disease Prediction — Full Project Guide

### KNN + K-Fold + Model Comparison (scikit-learn)

This project is a complete machine learning pipeline for predicting heart disease, starting from exploratory data analysis to training multiple models and comparing their performance.

## 🗃️ Dataset Info (heart.csv columns)

| Column | Type | Description |
|---|---|---|
| Age | int | Patient age |
| Sex | categorical | M / F |
| ChestPainType | categorical | ATA, NAP, ASY, TA |
| RestingBP | int | Resting blood pressure |
| Cholesterol | int | Serum cholesterol |
| FastingBS | int | Fasting blood sugar > 120 mg/dl (1=True) |
| RestingECG | categorical | Normal, ST, LVH |
| MaxHR | int | Max heart rate achieved |
| ExerciseAngina | categorical | Y / N |
| Oldpeak | float | ST depression |
| ST_Slope | categorical | Up, Flat, Down |
| **HeartDisease** | **int** | **Target — 0: No, 1: Yes** |

## 📊 Key Concepts Summary

| Term | What it means in this project |
|---|---|
| **KNN** | Predicts based on the K most similar patients in the training data |
| **K-Fold** | Trains/tests 10 times on different data splits → reliable performance |
| **Stratified** | Each fold has same % of heart disease patients as the full dataset |
| **Recall / Sensitivity** | Of all real heart disease patients, how many did we catch? (Most important here) |
| **Specificity** | Of all healthy patients, how many did we correctly say are healthy? |
| **Precision** | Of all patients we flagged as sick, how many were actually sick? |
| **False Negative** | We said patient is healthy → they actually have heart disease → DANGEROUS |
| **ROC-AUC** | Overall model quality — 1.0 = perfect, 0.5 = random guess |

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset from Kaggle**
   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and save it as `data/heart.csv` within the project folder.

3. **Run notebooks in order**
   Start your Jupyter Notebook server and run them sequentially:
   ```bash
   jupyter notebook
   ```
   Open: `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_knn_model.ipynb` → `04_kfold_evaluation.ipynb` → `05_model_comparison.ipynb`
