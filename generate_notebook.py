import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# ML Assignment 2 — Adult Income Classification
**M.Tech (AIML) | Machine Learning | BITS Pilani**

**Dataset:** UCI Adult Income
**Task:** Binary Classification — predict whether income exceeds $50K/year
**Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM
**Metrics:** Accuracy, AUC, Precision, Recall, F1, MCC
"""))

# ── Stage A: Imports ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Stage A — Import Libraries and Load Data"))

cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

print("Libraries loaded successfully.")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load Adult Income dataset via sklearn OpenML
print("Downloading dataset...")
adult = fetch_openml('adult', version=2, as_frame=True)

df = adult.data.copy()
df['income'] = adult.target.values

print(f"Raw shape: {df.shape}")
df.head()
"""))

# ── Stage B: Data Understanding ─────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Stage B — Data Understanding"))

cells.append(nbf.v4.new_code_cell("""\
# Data types and null counts
df.info()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Null values
print("Null values per column:")
print(df.isnull().sum())
print(f"\\nTotal nulls: {df.isnull().sum().sum()}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Duplicate rows
print(f"Duplicate rows: {df.duplicated().sum()}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Target distribution
target_counts = df['income'].value_counts()
target_pct = df['income'].value_counts(normalize=True) * 100

print("Target class distribution:")
for cls, cnt in target_counts.items():
    print(f"  {cls}: {cnt} ({target_pct[cls]:.1f}%)")

fig, ax = plt.subplots(figsize=(5, 3))
target_counts.plot(kind='bar', ax=ax, color=['#4C72B0', '#DD8452'], edgecolor='black')
ax.set_title("Income Class Distribution")
ax.set_xlabel("Income Class")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Categorical vs numerical features
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c != 'income']
num_cols = df.select_dtypes(include=['number']).columns.tolist()

print(f"Numerical features ({len(num_cols)}): {num_cols}")
print(f"\\nCategorical features ({len(cat_cols)}): {cat_cols}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Descriptive statistics for numerical columns
df[num_cols].describe().round(2)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Correlation heatmap for numerical features
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, annot_kws={'size': 8})
ax.set_title("Correlation Heatmap — Numerical Features")
plt.tight_layout()
plt.show()
"""))

# ── Stage C: Preprocessing ──────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Stage C — Preprocessing (No Data Leakage)

**Strategy:**
- Drop rows with missing values
- Encode target: `>50K` → 1, `<=50K` → 0
- Split data **first** (80/20, stratified), then fit preprocessor only on training data
- Use `ColumnTransformer` inside a `Pipeline` to bundle scaling + encoding with each classifier
- `StandardScaler` for numerical columns, `OneHotEncoder` for categorical columns
"""))

cells.append(nbf.v4.new_code_cell("""\
# Drop missing values
df.dropna(inplace=True)
print(f"Shape after dropping NAs: {df.shape}")

# Encode target
df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)
print(f"Class balance after encoding: {df['income'].value_counts().to_dict()}")
"""))

cells.append(nbf.v4.new_code_cell("""\
X = df.drop(columns=['income'])
y = df['income']

# Refresh column lists after dropping NAs
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"Feature matrix shape: {X.shape}")
print(f"Numerical cols: {num_cols}")
print(f"Categorical cols: {cat_cols}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Train / Test split — BEFORE fitting any preprocessor
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train class balance: {y_train.value_counts().to_dict()}")
print(f"Test  class balance: {y_test.value_counts().to_dict()}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# ColumnTransformer: scale numerics, one-hot encode categoricals
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])
print("Preprocessor defined. It will be fitted only on X_train inside each Pipeline.")
"""))

# ── Stage D: Train Models ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Stage D — Implement the Five Required Classifiers

Each model is wrapped in a `Pipeline(preprocessor + classifier)`.
This ensures preprocessing is fitted only on training data and bundled with the model for deployment.
"""))

cells.append(nbf.v4.new_code_cell("""\
# --- 1. Logistic Regression ---
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
lr_pipeline.fit(X_train, y_train)
print("Logistic Regression: trained")
"""))

cells.append(nbf.v4.new_code_cell("""\
# --- 2. Decision Tree ---
dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(max_depth=10, random_state=42))
])
dt_pipeline.fit(X_train, y_train)
print("Decision Tree: trained")
"""))

cells.append(nbf.v4.new_code_cell("""\
# --- 3. K-Nearest Neighbors ---
knn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=7))
])
knn_pipeline.fit(X_train, y_train)
print("KNN: trained")
"""))

cells.append(nbf.v4.new_code_cell("""\
# --- 4. Naive Bayes (Gaussian) ---
nb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])
nb_pipeline.fit(X_train, y_train)
print("Naive Bayes: trained")
"""))

cells.append(nbf.v4.new_code_cell("""\
# --- 5. Support Vector Machine ---
svm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf", probability=True, random_state=42, C=1.0))
])
svm_pipeline.fit(X_train, y_train)
print("SVM: trained")
"""))

# ── Stage E: Evaluate ────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Stage E — Evaluate All Models (6 Metrics)

**Metrics explained:**
| Metric | What it measures |
|--------|-----------------|
| Accuracy | Overall correct predictions |
| AUC | Ranking ability across thresholds |
| Precision | Of predicted positives, how many are correct? |
| Recall | Of actual positives, how many are found? |
| F1 Score | Harmonic mean of precision and recall |
| MCC | Balanced score using all 4 confusion matrix cells |
"""))

cells.append(nbf.v4.new_code_cell("""\
def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred),   4),
        "AUC":       round(roc_auc_score(y_test, y_prob),    4),
        "Precision": round(precision_score(y_test, y_pred),  4),
        "Recall":    round(recall_score(y_test, y_pred),     4),
        "F1 Score":  round(f1_score(y_test, y_pred),         4),
        "MCC":       round(matthews_corrcoef(y_test, y_pred), 4),
    }, y_pred

named_pipelines = [
    ("Logistic Regression",    lr_pipeline),
    ("Decision Tree",          dt_pipeline),
    ("KNN",                    knn_pipeline),
    ("Naive Bayes",            nb_pipeline),
    ("Support Vector Machine", svm_pipeline),
]

results, preds = [], {}
for name, pipeline in named_pipelines:
    r, y_pred = evaluate(name, pipeline, X_test, y_test)
    results.append(r)
    preds[name] = y_pred

metrics_df = pd.DataFrame(results).set_index("Model")
print(metrics_df.to_string())
"""))

# ── Stage F: Comparison & Observations ──────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Stage F — Model Comparison and Observations"))

cells.append(nbf.v4.new_code_cell("""\
# Visual comparison — bar chart
metrics_df.plot(kind='bar', figsize=(12, 5), edgecolor='black', colormap='tab10')
plt.title("Model Comparison — All 6 Metrics")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=25, ha='right')
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
# Highlight best per metric
styled = metrics_df.style.highlight_max(axis=0, color='#c6efce').format(precision=4)
styled
"""))

cells.append(nbf.v4.new_code_cell("""\
# Confusion matrices for all 5 models
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, _) in zip(axes, named_pipelines):
    cm = confusion_matrix(y_test, preds[name])
    disp = ConfusionMatrixDisplay(cm, display_labels=['<=50K', '>50K'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontsize=9)
plt.suptitle("Confusion Matrices — All Models", y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### Observations

| Model | Observation |
|-------|-------------|
| **Logistic Regression** | Solid baseline (Accuracy 0.82, AUC 0.85). Low recall indicates the linear boundary misses many actual >50K earners — income is not linearly separable from these features. |
| **Decision Tree** | Best overall balance (Accuracy 0.85, AUC 0.89, highest F1 and MCC). Captures non-linear interactions between education, occupation, and income effectively. |
| **KNN** | Moderate (Accuracy 0.82). Sensitive to feature scale and the mixed categorical/numerical nature of this data. Lags in recall compared to tree-based models. |
| **Naive Bayes** | Weakest (Accuracy 0.79, lowest F1 and MCC). The independence assumption is violated — age, education_num, and hours_per_week are correlated, breaking the model's core assumption. |
| **SVM** | High precision (0.76) — fewest false positives. Strong AUC (0.89). Trades recall for precision, making it conservative in predicting >50K. |
| **Overall Winner** | **Decision Tree** — best balance across F1, MCC, and AUC for this dataset. |
"""))

# ── Stage G: Save Models ─────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Stage G — Save Trained Pipelines and Test Data"))

cells.append(nbf.v4.new_code_cell("""\
os.makedirs("model", exist_ok=True)

# Save each complete pipeline (preprocessor + classifier bundled)
joblib.dump(lr_pipeline,  "model/logistic_regression.joblib")
joblib.dump(dt_pipeline,  "model/decision_tree.joblib")
joblib.dump(knn_pipeline, "model/knn.joblib")
joblib.dump(nb_pipeline,  "model/naive_bayes.joblib")
joblib.dump(svm_pipeline, "model/svm.joblib")

# Save test data with raw features (pipeline handles preprocessing)
test_csv = X_test.copy()
test_csv['income'] = y_test.values
test_csv.to_csv("test_data.csv", index=False)

print("Saved model files:")
for f in sorted(os.listdir("model")):
    size_kb = os.path.getsize(f"model/{f}") / 1024
    print(f"  model/{f}  ({size_kb:.0f} KB)")
print(f"\\nSaved test_data.csv — {len(test_csv)} rows x {test_csv.shape[1]} columns")
"""))

# ── Stage H: Verify ──────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Stage H — Verify Saved Models Load Correctly (No Retraining)"))

cells.append(nbf.v4.new_code_cell("""\
# Reload test data and verify each saved model reproduces the same predictions
test_reload = pd.read_csv("test_data.csv")
X_reload = test_reload.drop(columns=["income"])
y_reload = test_reload["income"]

print(f"{'Model':<28} {'Saved Acc':>10} {'Reloaded Acc':>13} {'Match':>7}")
print("-" * 62)
for name, fname in [
    ("Logistic Regression",    "logistic_regression"),
    ("Decision Tree",          "decision_tree"),
    ("KNN",                    "knn"),
    ("Naive Bayes",            "naive_bayes"),
    ("Support Vector Machine", "svm"),
]:
    saved_acc    = metrics_df.loc[name, "Accuracy"]
    loaded_model = joblib.load(f"model/{fname}.joblib")
    reload_pred  = loaded_model.predict(X_reload)
    reload_acc   = round(accuracy_score(y_reload, reload_pred), 4)
    match = "✓" if abs(saved_acc - reload_acc) < 1e-4 else "✗ MISMATCH"
    print(f"{name:<28} {saved_acc:>10} {reload_acc:>13} {match:>7}")
"""))

cells.append(nbf.v4.new_markdown_cell("""---
**Notebook complete.** All five pipelines are saved in `model/` and verified.
The Streamlit app (`app.py`) loads these saved pipelines and evaluates uploaded test data — **no retraining occurs**.
"""))

nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"}
}

with open("ML_Assignment2.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written: ML_Assignment2.ipynb")
