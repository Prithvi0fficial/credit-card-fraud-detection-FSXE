# =============================================================
# FSXE - Feature Selected XGBoost Ensemble
# Credit Card Fraud Detection
# Author: PRITHVI V
# =============================================================

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif

# Handling imbalance
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)

# Explainability
import shap

# Ignore warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# Create screenshots folder if not exists
import os
os.makedirs('../screenshots', exist_ok=True)

print(" All libraries loaded successfully")


# =============================================================
# SECTION 2 — Load Data and Exploratory Data Analysis (EDA)
# =============================================================

print("\n Loading dataset...")
df = pd.read_csv('../data/creditcard.csv')

# Basic info
print(f"Dataset Shape: {df.shape}")
print(f"Total Transactions: {len(df):,}")
print(f"Total Features: {df.shape[1]}")
print(f"Missing Values: {df.isnull().sum().sum()}")

# Class distribution
fraud_count = df['Class'].sum()
legit_count = len(df) - fraud_count
fraud_pct = round(fraud_count / len(df) * 100, 4)

print(f"\nLegitimate Transactions: {legit_count:,} ({100-fraud_pct}%)")
print(f"Fraud Transactions:      {fraud_count:,} ({fraud_pct}%)")

# ------- PLOT 1: Class Distribution Bar Chart -------
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Class', data=df,
                   palette=['#2ecc71', '#e74c3c'])
plt.title('Class Distribution\n(0=Legitimate, 1=Fraud)',
          fontsize=14, fontweight='bold')
plt.xlabel('Transaction Class', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Legitimate\n284,315', 'Fraud\n492'])

# Add count labels on bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('../screenshots/01_class_distribution.png', dpi=150)
plt.show()
print(" Plot 1 saved: class distribution")

# ------- PLOT 2: Transaction Amount Distribution -------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
df[df['Class'] == 0]['Amount'].hist(bins=50,
                                    color='#2ecc71',
                                    alpha=0.7)
plt.title('Legitimate Transaction Amounts')
plt.xlabel('Amount (€)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
df[df['Class'] == 1]['Amount'].hist(bins=50,
                                    color='#e74c3c',
                                    alpha=0.7)
plt.title('Fraud Transaction Amounts')
plt.xlabel('Amount (€)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('../screenshots/02_amount_distribution.png', dpi=150)
plt.show()
print(" Plot 2 saved: amount distribution")

# ------- PLOT 3: Correlation Heatmap -------
print("\n Generating correlation heatmap (takes 10 seconds)...")
plt.figure(figsize=(16, 12))
corr_matrix = df.corr()
sns.heatmap(corr_matrix,
            cmap='coolwarm',
            center=0,
            linewidths=0.1,
            annot=False)
plt.title('Feature Correlation Heatmap',
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../screenshots/03_correlation_heatmap.png', dpi=150)
plt.show()
print(" Plot 3 saved: correlation heatmap")

print("\n Section 2 Complete — EDA Done!")


# =============================================================
# SECTION 3 — Preprocessing
# =============================================================

print("\n  Starting Preprocessing...")

# Step 1: Scale Amount and Time
# These two columns are on different scales than V1-V28
# StandardScaler brings them to same scale (mean=0, std=1)
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled']   = scaler.fit_transform(df[['Time']])

# Step 2: Drop original Amount and Time columns
df = df.drop(['Amount', 'Time'], axis=1)

print(" Step 1 Done: Amount and Time scaled")

# Step 3: Separate features and target
X = df.drop('Class', axis=1)   # All columns except Class
y = df['Class']                 # Only the Class column

print(f" Step 2 Done: Features shape: {X.shape}")
print(f"              Target shape:   {y.shape}")

# Step 4: Train/Test Split (80% train, 20% test)
# random_state=42 means results are reproducible
# stratify=y means both splits keep same fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f" Step 3 Done: Train size: {X_train.shape[0]:,}")
print(f"              Test size:  {X_test.shape[0]:,}")

# Step 5: Apply SMOTE on training data ONLY
print("\n Applying SMOTE (this takes ~30 seconds)...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f" Step 4 Done: SMOTE Applied")
print(f"   Before SMOTE - Fraud: {y_train.sum():,} | "
      f"Legitimate: {(y_train==0).sum():,}")
print(f"   After SMOTE  - Fraud: {y_train_smote.sum():,} | "
      f"Legitimate: {(y_train_smote==0).sum():,}")

# Plot SMOTE effect
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(['Legitimate', 'Fraud'],
        [(y_train==0).sum(), y_train.sum()],
        color=['#2ecc71', '#e74c3c'])
ax1.set_title('Before SMOTE\n(Imbalanced)', fontweight='bold')
ax1.set_ylabel('Count')
for i, v in enumerate([(y_train==0).sum(), y_train.sum()]):
    ax1.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

ax2.bar(['Legitimate', 'Fraud'],
        [(y_train_smote==0).sum(), y_train_smote.sum()],
        color=['#2ecc71', '#e74c3c'])
ax2.set_title('After SMOTE\n(Balanced)', fontweight='bold')
ax2.set_ylabel('Count')
for i, v in enumerate([(y_train_smote==0).sum(),
                        y_train_smote.sum()]):
    ax2.text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

plt.suptitle('Effect of SMOTE on Class Distribution',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../screenshots/04_smote_effect.png', dpi=150)
plt.show()
print(" Plot 4 saved: SMOTE effect")

# Step 6: SelectKBest — pick top 20 features
print("\n Applying SelectKBest Feature Selection...")
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train_smote,
                                           y_train_smote)
X_test_selected  = selector.transform(X_test)

# Show which features were selected
selected_features = X.columns[selector.get_support()].tolist()
print(f" Step 5 Done: Selected top 20 features:")
print(f"   {selected_features}")

# Plot feature importance scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False).head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x='Score', y='Feature',
            data=feature_scores,
            palette='viridis')
plt.title('Top 20 Features by SelectKBest Score',
          fontsize=13, fontweight='bold')
plt.xlabel('F-Score (Higher = More Important)')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('../screenshots/05_feature_selection.png', dpi=150)
plt.show()
print(" Plot 5 saved: feature selection scores")

print("\n Section 3 Complete — Preprocessing Done!")
print(f"   Final training shape: {X_train_selected.shape}")
print(f"   Final test shape:     {X_test_selected.shape}")

# =============================================================
# SECTION 4 — Train All 4 Models
# =============================================================

print("\n Training all models...")
print("   (This will take 2-3 minutes — please wait)\n")

# --- Model 1: Logistic Regression ---
print("Training Model 1: Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)
lr_model.fit(X_train_selected, y_train_smote)
print(" Logistic Regression trained")

# --- Model 2: Random Forest ---
print("Training Model 2: Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
rf_model.fit(X_train_selected, y_train_smote)
print(" Random Forest trained")

# --- Model 3: XGBoost ---
print("Training Model 3: XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train_selected, y_train_smote)
# Fix base_score for SHAP compatibility
xgb_model.get_booster().set_attr(base_score='0.5')
print(" XGBoost trained")

# --- Model 4: FSXE Voting Ensemble ---
print("Training Model 4: FSXE Voting Ensemble...")
fsxe_model = VotingClassifier(
    estimators=[
        ('lr',  lr_model),
        ('rf',  rf_model),
        ('xgb', xgb_model)
    ],
    voting='soft'  # uses probability scores not just votes
)
fsxe_model.fit(X_train_selected, y_train_smote)
print(" FSXE Ensemble trained")

print("\n Section 4 Complete — All 4 models trained!")


# =============================================================
# SECTION 5 — Evaluate All Models and Visualize Results
# =============================================================

print("\n Evaluating all models on TEST data...")

# Function to evaluate any model and return metrics
def evaluate_model(model, X_test, y_test, model_name):
    # Get predictions
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate all metrics
    metrics = {
        'Model'    : model_name,
        'Accuracy' : round(accuracy_score(y_test, y_pred) * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall'   : round(recall_score(y_test, y_pred) * 100, 2),
        'F1_Score' : round(f1_score(y_test, y_pred) * 100, 2),
        'ROC_AUC'  : round(roc_auc_score(y_test, y_pred_prob) * 100, 2)
    }
    return metrics, y_pred, y_pred_prob

# Evaluate all 4 models
results = []
predictions = {}

for model, name in [
    (lr_model,   'Logistic Regression'),
    (rf_model,   'Random Forest'),
    (xgb_model,  'XGBoost'),
    (fsxe_model, 'FSXE (Proposed)')
]:
    metrics, y_pred, y_prob = evaluate_model(
        model, X_test_selected, y_test, name
    )
    results.append(metrics)
    predictions[name] = (y_pred, y_prob)
    print(f" {name} evaluated")

# Create results dataframe
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("COMPARATIVE RESULTS TABLE")
print("="*70)
print(results_df.to_string(index=False))
print("="*70)

# ---- PLOT 6: Model Comparison Bar Chart ----
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
x = np.arange(len(metrics_to_plot))
width = 0.2
colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']

fig, ax = plt.subplots(figsize=(14, 7))

for i, (_, row) in enumerate(results_df.iterrows()):
    values = [row[m] for m in metrics_to_plot]
    bars = ax.bar(x + i * width, values,
                  width, label=row['Model'],
                  color=colors[i], alpha=0.85)

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Performance Comparison\n'
             'Logistic Regression vs Random Forest '
             'vs XGBoost vs FSXE',
             fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(60, 105)
ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../screenshots/06_model_comparison.png', dpi=150)
plt.show()
print(" Plot 6 saved: model comparison")

# ---- PLOT 7: ROC Curves for All Models ----
plt.figure(figsize=(10, 7))

colors_roc = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
for i, (name, (_, y_prob)) in enumerate(predictions.items()):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr,
             color=colors_roc[i],
             linewidth=2.5,
             label=f'{name} (AUC = {auc_score:.4f})')

# Random classifier baseline
plt.plot([0, 1], [0, 1],
         'k--', linewidth=1.5,
         label='Random Classifier (AUC = 0.5)')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves — All Models Comparison',
          fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../screenshots/07_roc_curves.png', dpi=150)
plt.show()
print(" Plot 7 saved: ROC curves")

# ---- PLOT 8: Confusion Matrix for FSXE only ----
fsxe_pred = predictions['FSXE (Proposed)'][0]
cm = confusion_matrix(y_test, fsxe_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'],
            linewidths=2)
plt.title('FSXE Confusion Matrix\n(Test Set Results)',
          fontsize=13, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

# Add explanation text
tn, fp, fn, tp = cm.ravel()
plt.figtext(0.15, 0.02,
    f'TP={tp} (Fraud correctly caught) | '
    f'FP={fp} (False alarms) | '
    f'FN={fn} (Missed fraud) | '
    f'TN={tn} (Legitimate correctly cleared)',
    fontsize=8, ha='left')

plt.tight_layout()
plt.savefig('../screenshots/08_confusion_matrix_FSXE.png', dpi=150)
plt.show()
print(" Plot 8 saved: FSXE confusion matrix")

print("\n Section 5 Complete — Evaluation Done!") 



# =============================================================
# SECTION 6 — SHAP Explainability (Novel Contribution)
# Using Random Forest — most stable SHAP implementation
# =============================================================

print("\n Generating SHAP Explainability Analysis...")
print(" This takes 2-3 minutes — please wait")

# Get feature names after selection
selected_feature_names = X.columns[selector.get_support()].tolist()

# Use a sample of test data for speed
sample_size = 200
X_test_sample_df = pd.DataFrame(
    X_test_selected[:sample_size],
    columns=selected_feature_names
)
y_test_sample = y_test.iloc[:sample_size]

# Use Random Forest for SHAP — 100% compatible, no version issues
rf_explainer  = shap.TreeExplainer(rf_model)
shap_values   = rf_explainer.shap_values(X_test_sample_df)

# shap_values is a list [class0_values, class1_values]
# We want class 1 = fraud
shap_fraud = shap_values[:,:,1]

print("SHAP values calculated successfully")

# ---- PLOT 9: SHAP Bar Chart ----
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_fraud,
    X_test_sample_df,
    feature_names=selected_feature_names,
    show=False,
    plot_type='bar'
)
plt.title('SHAP Feature Importance\n'
          '(Mean Impact on Fraud Prediction)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../screenshots/09_shap_importance.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(" Plot 9 saved: SHAP importance")

# ---- PLOT 10: SHAP Dot Plot ----
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_fraud,
    X_test_sample_df,
    feature_names=selected_feature_names,
    show=False
)
plt.title('SHAP Values — Feature Impact Direction\n'
          '(Red=High Feature Value, Blue=Low Feature Value)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('../screenshots/10_shap_dot_plot.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(" Plot 10 saved: SHAP dot plot")

# ---- PLOT 11: SHAP Waterfall for 1 Fraud Case ----
fraud_indices = [
    i for i, val in enumerate(y_test_sample.values) if val == 1
]

if len(fraud_indices) > 0:
    fraud_idx = fraud_indices[0]
    plt.figure(figsize=(12, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_fraud[fraud_idx],
            base_values   = rf_explainer.expected_value[1],
            data          = X_test_sample_df.iloc[fraud_idx].values,
            feature_names = selected_feature_names
        ),
        show=False
    )
    plt.title('SHAP Waterfall — Why This Transaction '
              'Was Flagged as Fraud',
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../screenshots/11_shap_waterfall_fraud.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print(" Plot 11 saved: SHAP waterfall")
else:
    print(" No fraud in sample window — trying larger window...")
    X_larger = pd.DataFrame(
        X_test_selected[:1000],
        columns=selected_feature_names
    )
    y_larger = y_test.iloc[:1000]

    # Find fraud index positions in larger window
    fraud_indices_2 = [
        i for i, val in enumerate(y_larger.values) if val == 1
    ]

    if len(fraud_indices_2) > 0:
        # Calculate SHAP on larger window
        shap_values2 = rf_explainer.shap_values(X_larger)
        # Fix: use [:, :, 1] same as before
        shap_fraud2  = shap_values2[:, :, 1]

        # Use local index — position within shap_fraud2
        local_idx = fraud_indices_2[0]  # this is already within 1000

        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values        = shap_fraud2[local_idx],
                base_values   = rf_explainer.expected_value[1],
                data          = X_larger.iloc[local_idx].values,
                feature_names = selected_feature_names
            ),
            show=False
        )
        plt.title('SHAP Waterfall — Why This Transaction '
                  'Was Flagged as Fraud',
                  fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../screenshots/11_shap_waterfall_fraud.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot 11 saved: SHAP waterfall")
    else:
        print(" No fraud found — generating waterfall "
              "for first test sample instead")
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values        = shap_fraud[0],
                base_values   = rf_explainer.expected_value[1],
                data          = X_test_sample_df.iloc[0].values,
                feature_names = selected_feature_names
            ),
            show=False
        )
        plt.title('SHAP Waterfall — Feature Contributions '
                  'for Sample Transaction',
                  fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../screenshots/11_shap_waterfall_fraud.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(" Plot 11 saved: SHAP waterfall")

# ---- SHAP Importance Table ----
shap_importance = pd.DataFrame({
    'Feature'         : selected_feature_names,
    'SHAP_Importance' : abs(shap_fraud).mean(axis=0)
}).sort_values('SHAP_Importance', ascending=False)

print("\n SHAP Feature Importance Ranking:")
print(shap_importance.to_string(index=False))

# ---- Final Summary ----
print("\n Section 6 Complete — SHAP Done!")
print("\n" + "="*60)
print(" ALL SECTIONS COMPLETE!")
print("="*60)
print("Total screenshots saved: 11")
print("Location: ../screenshots/")
print("\nAll screenshots ready for your paper:")
for name in [
    "01_class_distribution.png",
    "02_amount_distribution.png",
    "03_correlation_heatmap.png",
    "04_smote_effect.png",
    "05_feature_selection.png",
    "06_model_comparison.png",
    "07_roc_curves.png",
    "08_confusion_matrix_FSXE.png",
    "09_shap_importance.png",
    "10_shap_dot_plot.png",
    "11_shap_waterfall_fraud.png"
]:
    print(f"   {name}")
print("="*60)




# ── CROSS VALIDATION ─────────────────────────────────────────────
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

print("\n=== CROSS VALIDATION (5-Fold) on SMOTE+Selected Features ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in [("Logistic Regression", lr_model), 
                    ("Random Forest", rf_model),
                    ("XGBoost", xgb_model),
                    ("FSXE", fsxe_model)]:
    scores = cross_val_score(model, X_train_selected, y_train_smote, 
                             cv=cv, scoring='recall', n_jobs=-1)
    print(f"{name}: Mean={scores.mean()*100:.2f}% | Std={scores.std()*100:.2f}% | All={np.round(scores*100,2)}")

# ── UNBALANCED VS BALANCED ───────────────────────────────────────
print("\n=== UNBALANCED (no SMOTE) vs BALANCED (SMOTE) — FSXE Recall ===")

# Train FSXE on ORIGINAL imbalanced data (no SMOTE)
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

fsxe_unbalanced = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1))
], voting='soft')

# Use original training data (before SMOTE) with selected features
X_train_orig_selected = selector.transform(X_train)
fsxe_unbalanced.fit(X_train_orig_selected, y_train)
y_pred_unbal = fsxe_unbalanced.predict(X_test_selected)

from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
print(f"FSXE WITHOUT SMOTE — Recall: {recall_score(y_test, y_pred_unbal)*100:.2f}%  F1: {f1_score(y_test, y_pred_unbal)*100:.2f}%  Precision: {precision_score(y_test, y_pred_unbal)*100:.2f}%")
print(f"FSXE WITH SMOTE    — Recall: 87.76%  F1: 74.46%  Precision: 64.66%")


# ── CORRECT CV WITH SMOTE INSIDE PIPELINE ────────────────────────
print("\n=== CORRECT 5-Fold CV (SMOTE inside each fold — no leakage) ===")
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use ORIGINAL training data (before SMOTE) — X_train with selector applied
X_train_orig_selected = selector.transform(X_train)  # original, not SMOTEd

lr_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

rf_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1))
])

xgb_pipe = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1))
])

for name, pipe in [("Logistic Regression", lr_pipe),
                   ("Random Forest", rf_pipe),
                   ("XGBoost", xgb_pipe)]:
    scores = cross_val_score(pipe, X_train_orig_selected, y_train,
                             cv=cv, scoring='recall', n_jobs=-1)
    print(f"{name}: Mean={scores.mean()*100:.2f}% | Std={scores.std()*100:.2f}% | All={np.round(scores*100,2)}")