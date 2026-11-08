
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification Dashboard")

st.write(
    "Machine Learning Classification using the UCI Dry Bean Dataset"
)

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength",
    "MinorAxisLength", "AspectRatio", "Eccentricity",
    "ConvexArea", "EquivDiameter", "Extent",
    "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2",
    "ShapeFactor3", "ShapeFactor4"
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_models():
    return {
        name: joblib.load(os.path.join("model", filename))
        for name, filename in MODEL_FILES.items()
    }

uploaded_file = st.file_uploader(
    "Upload your labelled test CSV",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    missing = [
        col for col in FEATURES + ["Class"]
        if col not in data.columns
    ]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    if data[FEATURES + ["Class"]].isnull().any().any():
        st.error("Please upload data without missing values.")
        st.stop()

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = data["Class"].astype(str)

    if X.isnull().any().any() or not np.isfinite(X.to_numpy()).all():
        st.error("Input features must contain valid finite numbers.")
        st.stop()

    try:
        models = load_models()
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

    selected = st.selectbox(
        "Select Classification Model",
        list(models.keys())
    )

    model = models[selected]

    if not set(y.unique()).issubset(set(model.classes_)):
        st.error("Unknown class labels in uploaded data.")
        st.stop()

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y, predictions, average="weighted", zero_division=0
    )

    recall = recall_score(
        y, predictions, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y, predictions, average="weighted", zero_division=0
    )

    mcc = matthews_corrcoef(y, predictions)

    try:
        auc = roc_auc_score(
            y, probabilities,
            labels=model.classes_,
            multi_class="ovr",
            average="weighted"
        )
    except ValueError:
        auc = np.nan

    st.header("Evaluation Metrics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("AUC", f"{auc:.4f}" if np.isfinite(auc) else "N/A")
    c3.metric("Precision", f"{precision:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", f"{recall:.4f}")
    c5.metric("F1 Score", f"{f1:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    st.header("Confusion Matrix")

    labels = model.classes_
    cm = confusion_matrix(y, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=labels, yticklabels=labels, ax=ax
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    st.pyplot(fig)
    plt.close(fig)

    st.header("Classification Report")

    report = classification_report(
        y, predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(pd.DataFrame(report).transpose().round(4))

    st.header("Comparison of All Five Models")

    results = []

    for name, clf in models.items():
        pred = clf.predict(X)
        prob = clf.predict_proba(X)

        results.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(
                y, prob,
                labels=clf.classes_,
                multi_class="ovr",
                average="weighted"
            ),
            "Precision": precision_score(
                y, pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                y, pred, average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                y, pred, average="weighted", zero_division=0
            ),
            "MCC": matthews_corrcoef(y, pred)
        })

    comparison = pd.DataFrame(results)

    st.dataframe(
        comparison.style.format({
            col: "{:.4f}" for col in comparison.columns
            if col != "ML Model Name"
        })
    )

    winner = comparison.loc[comparison["Accuracy"].idxmax()]

    st.success(
        f"Best model: {winner['ML Model Name']} "
        f"with accuracy {winner['Accuracy']:.4f}"
    )

    st.download_button(
        "Download Comparison Results",
        comparison.to_csv(index=False),
        "model_comparison_results.csv",
        "text/csv"
    )

else:
    st.info("Upload test_data.csv to begin model evaluation.")
