import pandas as pd
import numpy as np
import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

# ── Load dataset ──────────────────────────────────────────────────────────────
print("Downloading Adult Income dataset via OpenML...")
adult = fetch_openml('adult', version=2, as_frame=True)
df = adult.data.copy()
df['income'] = adult.target.values
df.dropna(inplace=True)
print(f"Dataset shape after dropping NAs: {df.shape}")

# ── Preprocess ────────────────────────────────────────────────────────────────
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

CAT_COLS = df.select_dtypes(include=['object', 'category']).columns.tolist()
CAT_COLS = [c for c in CAT_COLS if c != 'income']

le = LabelEncoder()
for col in CAT_COLS:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Save test data (preprocessed) for Streamlit upload
test_csv = X_test.copy()
test_csv['income'] = y_test.values
test_csv.to_csv('test_data.csv', index=False)
print(f"test_data.csv saved — {len(test_csv)} rows")

# ── Scale (for LR, KNN, NB, SVM) ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.joblib')

# Models that need scaled input
NEEDS_SCALING = {'logistic_regression', 'knn', 'naive_bayes', 'svm'}

MODELS = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    'decision_tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    'knn':                 KNeighborsClassifier(n_neighbors=7),
    'naive_bayes':         GaussianNB(),
    'svm':                 SVC(probability=True, kernel='rbf', random_state=42),
}

# ── Train & evaluate ──────────────────────────────────────────────────────────
print("\n{'Model':<25} {'Acc':>6} {'AUC':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'MCC':>6}")
print("-" * 65)

for name, model in MODELS.items():
    Xtr = X_train_sc if name in NEEDS_SCALING else X_train
    Xte = X_test_sc  if name in NEEDS_SCALING else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    metrics = {
        'accuracy':  round(accuracy_score(y_test, y_pred),  4),
        'auc':       round(roc_auc_score(y_test, y_prob),   4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall':    round(recall_score(y_test, y_pred),    4),
        'f1':        round(f1_score(y_test, y_pred),        4),
        'mcc':       round(matthews_corrcoef(y_test, y_pred), 4),
    }
    print(f"{name:<25} {metrics['accuracy']:>6} {metrics['auc']:>6} "
          f"{metrics['precision']:>6} {metrics['recall']:>6} "
          f"{metrics['f1']:>6} {metrics['mcc']:>6}")

    joblib.dump(model, f'model/{name}.joblib')

print("\nAll models saved to model/")
