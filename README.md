# Machine Learning Assignment 2 - Classification Models & Streamlit Deployment

## a. Problem Statement
The objective of this project is to classify dry beans into seven distinct registered varieties based on 16 geometric and shape features extracted from high-resolution images using five different machine learning classification algorithms (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, and Support Vector Machine), and to deploy the trained models on an interactive Streamlit web application.

## b. Dataset Description
* **Dataset Name:** UCI Dry Bean Dataset
* **Source:** UCI Machine Learning Repository
* **Problem Type:** Multi-class Classification (7 classes)
* **Number of Instances (Rows):** 13,611 (2,723 test instances in `test_data.csv`)
* **Number of Features (Columns):** 16 (`Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`, `AspectRatio`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`, `Solidity`, `Roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`)
* **Target Variable:** `Class` (7 bean varieties: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, and Sira)
* **Preprocessing Applied:** Missing value verification, Label Encoding for the multi-class target variable (`Class`), stratified train-test split (80:20 ratio), and feature scaling using `StandardScaler` to normalize morphological measurements across differing scales.

## c. GitHub Repository Link
* **GitHub Repository:** https://github.com/harisowmya1094-prog/Dry-Bean-Classification.git

### Repository Structure
```text
project-folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── scaler.pkl
```

## d. Models Used & Performance Evaluation

### 1. Model Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| **Decision Tree** | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| **KNN** | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| **Naive Bayes** | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| **Support Vector Machine** | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

### 2. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
| :--- | :--- |
| **Logistic Regression** | Performs very strongly as the second-best model with **92.07% Accuracy**, **0.9934 AUC**, and an **MCC of 0.9042**. Standardized morphological features allow linear decision boundaries to effectively separate most bean varieties, with minor confusion only between morphologically overlapping classes like Sira and Dermason. |
| **Decision Tree** | Achieves moderate performance (**89.20% Accuracy**, **0.9334 AUC**, and **0.8696 MCC**). While it captures non-linear feature interactions without requiring feature scaling, axis-aligned splits on continuous geometric measurements result in slightly lower generalization compared to margin-based classifiers. |
| **KNN** | Delivers strong classification performance (**91.66% Accuracy**, **0.9812 AUC**, and **0.8992 MCC**). Feature scaling with `StandardScaler` ensures that local Euclidean distances in the 16-dimensional feature space accurately group morphologically similar bean varieties. |
| **Naive Bayes** | Records the lowest performance among the five models (**76.39% Accuracy**, **0.7615 F1**, and **0.7154 MCC**), despite a relatively high **0.9644 AUC**. This drop occurs because the strong conditional independence assumption is violated by heavily correlated geometric features (`Area`, `Perimeter`, `ConvexArea`, and `EquivDiameter`). |
| **Support Vector Machine** | Achieves the highest performance across all six evaluation metrics (**92.21% Accuracy**, **0.9937 AUC**, **0.9224 Precision**, **0.9221 Recall**, **0.9222 F1**, and **0.9059 MCC**). Kernel mapping effectively handles non-linear class boundaries and collinearity among shape descriptors. |
| **Overall, Winner for Dataset** | **Support Vector Machine (SVM)** is the overall winner for the UCI Dry Bean dataset, achieving top scores across every metric (**92.21% Accuracy**, **0.9937 AUC**, and **0.9059 MCC**) and marginally outperforming Logistic Regression (**92.07%**) and KNN (**91.66%**). |