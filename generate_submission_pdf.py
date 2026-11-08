from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)
from reportlab.lib.enums import TA_CENTER

OUTPUT = "/Users/jessica.dsouza/Downloads/ML_Assignment2_Submission.pdf"

doc = SimpleDocTemplate(OUTPUT, pagesize=A4,
                        rightMargin=2*cm, leftMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

styles = getSampleStyleSheet()
title  = ParagraphStyle("t",  parent=styles["Title"],   fontSize=15, spaceAfter=4)
sub    = ParagraphStyle("s",  parent=styles["Title"],   fontSize=12, spaceAfter=6)
h1     = ParagraphStyle("h1", parent=styles["Heading1"],fontSize=12, spaceAfter=4,
                         textColor=colors.HexColor("#1a3c6b"))
h2     = ParagraphStyle("h2", parent=styles["Heading2"],fontSize=11, spaceAfter=3,
                         textColor=colors.HexColor("#2e6da4"))
body   = ParagraphStyle("b",  parent=styles["Normal"],  fontSize=10, spaceAfter=4, leading=15)
link   = ParagraphStyle("l",  parent=styles["Normal"],  fontSize=10, spaceAfter=4,
                         textColor=colors.HexColor("#0563C1"))
note   = ParagraphStyle("n",  parent=styles["Normal"],  fontSize=9,  spaceAfter=4,
                         textColor=colors.grey, leftIndent=10)
code   = ParagraphStyle("c",  parent=styles["Normal"],  fontSize=9,  spaceAfter=4,
                         fontName="Courier", leftIndent=10)

def HR(): return HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#aaaaaa"),
                             spaceAfter=6, spaceBefore=6)

def B(t): return f"<b>{t}</b>"

def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#dce6f0")),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f5f8fc")]),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("ALIGN",         (0,0),(0,-1), "LEFT"),
    ]))
    return t

story = []

# ── Header ────────────────────────────────────────────────────────────────────
story += [
    Paragraph("M.Tech (AIML) — Machine Learning", title),
    Paragraph("Assignment 2 — Submission", sub),
    Paragraph("Work Integrated Learning Programmes Division, BITS Pilani", note),
    HR(), Spacer(1, 0.3*cm),
]

# ── Section 1: Links ──────────────────────────────────────────────────────────
story.append(Paragraph("1. Submission Links", h1))

story.append(Paragraph(B("GitHub Repository Link:"), h2))
story.append(Paragraph(
    '<link href="https://github.com/JessicaNaomiDsouza/adult-income-classifier">'
    'https://github.com/JessicaNaomiDsouza/adult-income-classifier</link>', link))
story.append(Paragraph(
    "Repository contains: app.py, train.py, requirements.txt, README.md, "
    "test_data.csv, model/ (5 saved classifiers + scaler)", note))

story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(B("Live Streamlit App Link:"), h2))
story.append(Paragraph(
    '<link href="https://adult-income-classifier-sbgasuujptzpgftial2hcd.streamlit.app/">'
    'https://adult-income-classifier-sbgasuujptzpgftial2hcd.streamlit.app/</link>', link))

story += [HR(), Spacer(1, 0.2*cm)]

# ── Section 2: Screenshot placeholder ────────────────────────────────────────
story.append(Paragraph("2. BITS Virtual Lab Screenshot", h1))
story.append(Paragraph(
    "[Insert screenshot of assignment execution on BITS Virtual Lab here]", note))
story.append(Spacer(1, 3*cm))
story += [HR(), Spacer(1, 0.2*cm)]

# ── Section 3: README content ─────────────────────────────────────────────────
story.append(Paragraph("3. README.md Content", h1))

# a. Problem statement
story.append(Paragraph("a) Problem Statement", h2))
story.append(Paragraph(
    "Predict whether an individual's annual income exceeds $50,000 based on "
    "demographic and employment attributes. This is a binary classification problem "
    "(income ≤50K or >50K) using the UCI Adult Income dataset.", body))

# b. Dataset description
story.append(Paragraph("b) Dataset Description", h2))
desc_data = [
    ["Attribute", "Details"],
    ["Source",    "UCI Machine Learning Repository — Adult Income Dataset"],
    ["Features",  "14 input features (age, workclass, fnlwgt, education, education_num,\nmarital_status, occupation, relationship, race, sex,\ncapital_gain, capital_loss, hours_per_week, native_country)"],
    ["Target",    "income — binary: 0 = ≤50K, 1 = >50K"],
    ["Instances", "48,842 rows total; ~45,222 after removing missing values"],
    ["Task",      "Binary Classification"],
]
story.append(make_table(desc_data, col_widths=[4*cm, 12*cm]))

story.append(Spacer(1, 0.3*cm))

# c. GitHub link
story.append(Paragraph("c) GitHub Repository Link", h2))
story.append(Paragraph(
    '<link href="https://github.com/JessicaNaomiDsouza/adult-income-classifier">'
    'https://github.com/JessicaNaomiDsouza/adult-income-classifier</link>', link))

story.append(Spacer(1, 0.3*cm))

# d. Models & metrics
story.append(Paragraph("d) Models Used — Evaluation Metrics Comparison", h2))
metrics_data = [
    ["ML Model Name",           "Accuracy", "AUC",   "Precision", "Recall", "F1",    "MCC"],
    ["Logistic Regression",     "0.8172",   "0.8473","0.7075",    "0.4478", "0.5485","0.4592"],
    ["Decision Tree",           "0.8477",   "0.8926","0.7134",    "0.6441", "0.6770","0.5789"],
    ["KNN",                     "0.8239",   "0.8590","0.6693",    "0.5723", "0.6170","0.5061"],
    ["Naive Bayes",             "0.7947",   "0.8478","0.6717",    "0.3359", "0.4478","0.3692"],
    ["Support Vector Machine",  "0.8460",   "0.8913","0.7574",    "0.5571", "0.6420","0.5572"],
]
story.append(make_table(metrics_data,
    col_widths=[4.5*cm, 2*cm, 2*cm, 2.2*cm, 2*cm, 2*cm, 2*cm]))

story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("Observations on Model Performance", h2))
obs_data = [
    ["ML Model Name",           "Observation about model performance"],
    ["Logistic Regression",
     "Achieves 81.7% accuracy and AUC of 0.847. Low recall (0.448) means it "
     "misses many actual >50K earners. The linear decision boundary is "
     "insufficient for this dataset's complex relationships."],
    ["Decision Tree",
     "Best overall performer — highest accuracy (84.8%), AUC (0.893), F1 (0.677), "
     "and MCC (0.579). Captures non-linear relationships between income and "
     "features like education and occupation effectively."],
    ["KNN",
     "Moderate performance (82.4% accuracy). Sensitive to the high-dimensional "
     "mixed-type features; scaling helps but it still lags behind tree-based "
     "models in recall (0.572)."],
    ["Naive Bayes",
     "Lowest recall (0.336) and F1 (0.448). The independence assumption between "
     "features is strongly violated here — education and occupation are correlated "
     "— which limits its effectiveness."],
    ["Support Vector Machine",
     "Highest precision (0.757) — fewest false positives among all models. "
     "Good AUC (0.891) but moderate recall (0.557), meaning it is conservative "
     "in predicting the >50K class."],
    [B("Overall Winner"),
     B("Decision Tree — best balance of accuracy, AUC, F1, and MCC on this dataset.")],
]
story.append(make_table(obs_data, col_widths=[4.5*cm, 12*cm]))

story += [HR(), Spacer(1, 0.2*cm)]

# ── Section 4: Streamlit app link ─────────────────────────────────────────────
story.append(Paragraph("4. Live Streamlit App Link", h1))
story.append(Paragraph(
    '<link href="https://adult-income-classifier-sbgasuujptzpgftial2hcd.streamlit.app/">'
    'https://adult-income-classifier-sbgasuujptzpgftial2hcd.streamlit.app/</link>', link))

story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("Streamlit App Features:", h2))
features = [
    "• CSV upload option (test_data.csv) via sidebar",
    "• Model selection dropdown — choose from all 5 classifiers",
    "• Display of all 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)",
    "• Confusion matrix heatmap for the selected model",
    "• Classification report (precision/recall per class)",
    "• 'Run all models' button — shows comparison table for all 5 models at once",
]
for f in features:
    story.append(Paragraph(f, body))

doc.build(story)
print(f"PDF saved: {OUTPUT}")
