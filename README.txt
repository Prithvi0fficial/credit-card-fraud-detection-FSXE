Name: PRITHVI V
Email: prithvi8289@gmail.com
Assessment: Round 1 — Coding and Data Analysis Task
Internship: Python Development — Megaminds IT Services

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: FSXE — Feature Selected XGBoost Ensemble
Topic: Credit Card Fraud Detection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BASE PAPER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Title: Deep Learning for Credit Card Fraud Detection:
       A Review of Algorithms, Challenges, and Solutions
Authors: Ibomoiye Domor Mienye, Nobert Jere
Journal: IEEE Access, Volume 12, 2024
DOI: 10.1109/ACCESS.2024.3426955
Link: https://doi.org/10.1109/ACCESS.2024.3426955

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILES INSIDE THIS ZIP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
code/
  fraud_detection_FSXE.py     — Complete source code

screenshots/
  01_class_distribution.png
  02_amount_distribution.png
  03_correlation_heatmap.png
  04_smote_effect.png
  05_feature_selection.png
  06_model_comparison.png
  07_roc_curves.png
  08_confusion_matrix_FSXE.png
  09_shap_importance.png
  10_shap_dot_plot.png
  11_shap_waterfall_fraud.png

documents/
  research_paper_FSXE_FINAL.docx   — Full research paper
  case_study_FSXE.docx             — Case study document

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VIDEO PRESENTATION (15 minutes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Google Drive Link: https://drive.google.com/file/d/1PQvCZmbWitcnCC8skk6N5Hsbt5CuKWwR/view?usp=sharing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: European Credit Card Fraud Detection Dataset
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Note: Dataset not included in ZIP due to large file size (144MB)
      Download from Kaggle link above before running the code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN THE CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Download creditcard.csv from Kaggle link above
2. Place it in a folder called data/ next to the code/ folder
3. Install requirements:
   pip install pandas numpy scikit-learn xgboost
   pip install imbalanced-learn shap matplotlib seaborn
4. Run: python fraud_detection_FSXE.py
5. Screenshots will be saved to screenshots/ folder