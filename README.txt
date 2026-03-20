# Credit Card Fraud Detection — FSXE Algorithm

A novel ensemble algorithm combining Logistic Regression, Random Forest,
and XGBoost with SMOTE oversampling, SelectKBest feature selection,
and SHAP explainability for credit card fraud detection.

**Key result: 87.76% Recall — outperforms best deep learning model
(GRU, 79.59%) from the IEEE Access 2024 base paper.**

---

## What is FSXE?

FSXE (Feature Selected XGBoost Ensemble) solves three problems
simultaneously that existing deep learning models fail to address:

- Class imbalance (0.17% fraud) — solved with SMOTE oversampling
- Noisy input features — solved with SelectKBest (top 20 of 30 features)
- Lack of explainability — solved with SHAP values per transaction

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 97.31% | 5.57% | 91.84% | 10.50% | 96.63% |
| Random Forest | 99.93% | 78.43% | 81.63% | 80.00% | 96.71% |
| XGBoost | 99.88% | 60.28% | 86.73% | 71.13% | 97.84% |
| **FSXE (Proposed)** | **99.90%** | **64.66%** | **87.76%** | **74.46%** | **96.84%** |

FSXE outperforms all 6 deep learning models (MLP, CNN, RNN, LSTM,
BiLSTM, GRU) reported in the base paper on Recall — the primary
metric for fraud detection.

---

## Base Paper

> Mienye, I. D., & Jere, N. (2024). Deep learning for credit card
> fraud detection: A review of algorithms, challenges, and solutions.
> IEEE Access, 12, 96893–96910.
> DOI: 10.1109/ACCESS.2024.3426955

---

## Dataset

European Credit Card Fraud Detection Dataset — 284,807 transactions,
492 fraud cases (0.17%).

Download from Kaggle before running:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place as: `data/creditcard.csv`

---

## How to Run
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn

python code/fraud_detection_FSXE.py
```

Screenshots save automatically to `screenshots/`

---

## Key Visualizations

![Model Comparison](screenshots/06_model_comparison.png)
![SHAP Feature Importance](screenshots/09_shap_importance.png)
![Confusion Matrix](screenshots/08_confusion_matrix_FSXE.png)

---

## Tech Stack

Python · XGBoost · Scikit-learn · imbalanced-learn · SHAP ·
Matplotlib · Pandas · NumPy

---

## Documents

- [Research Paper](documents/research_paper_FSXE_FINAL.pdf)
- [Case Study](documents/case_study_FSXE.pdf)
