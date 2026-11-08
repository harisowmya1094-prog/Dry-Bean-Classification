# Dry Bean Variety Classification — ML Classification (Assignment 2)

**Live app:** https://dry-bean-classification-e84cyv9pnlt3qksdeesbbr.streamlit.app/

## a. Problem statement
The goal is to classify dry beans into one of seven registered varieties based on 16 geometric and shape features extracted from high-resolution images of grains. It is a multi-class classification problem (7 classes). I trained 5 classifiers on the same dataset, compared them using 6 evaluation metrics, and built an interactive Streamlit web application where the saved models can be tested on uploaded test data.

## b. Dataset description

| Item | Detail |
| :--- | :--- |
| **Name** | UCI Dry Bean Dataset |
| **Source** | UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/602/dry+bean+dataset |
| **Instances** | 13,611 grains (10,888 training rows, 2,723 test rows) |
| **Features** | 16 numeric input features + 1 target |
| **Target** | `Class` (7 bean varieties: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) |
| **Class balance** | Multi-class imbalanced (Dermason ~26.1%, Sira ~19.4%, Seker ~14.9%, Horoz ~14.2%, Cali ~12.0%, Barbunya ~9.7%, Bombay ~3.8%) |
| **Missing values** | None |

### Features

| Feature | Type | Description |
| :--- | :--- | :--- |
| `Area` | numeric | Area of a bean zone and the number of pixels within its boundaries |
| `Perimeter` | numeric | Circumference of a bean, defined as the length of its border |
| `MajorAxisLength`, `MinorAxisLength` | numeric | Distance between the ends of the longest line and the longest line perpendicular to the main axis |
| `AspectRatio` | numeric | Ratio between `MajorAxisLength` and `MinorAxisLength` |
| `Eccentricity` | numeric | Eccentricity of the ellipse having the same moments as the region |
| `ConvexArea` | numeric | Number of pixels in the smallest convex polygon that can contain the area of a bean seed |
| `EquivDiameter` | numeric | Diameter of a circle having the same area as a bean seed area |
| `Extent` | numeric | Ratio of the pixels in the bounding box to the bean area |
| `Solidity` | numeric | Ratio of the pixels in the convex shell to those found in beans |
| `Roundness` | numeric | Calculated with the formula $(4 \pi \times \text{Area}) / \text{Perimeter}^2$ |
| `Compactness` | numeric | Measures the roundness of an object ($\text{EquivDiameter} / \text{MajorAxisLength}$) |
| `ShapeFactor1` – `ShapeFactor4` | numeric | Four derived shape descriptors quantifying grain morphology and curvature |

### Preprocessing
1. Verified the dataset for missing values (none found across all 13,611 instances).
2. Encoded the categorical target column `Class` (7 bean varieties) using `LabelEncoder`.
3. Split the data 80/20 with stratification so that all 7 bean varieties maintain their exact class proportions in both sets. This gave 10,888 rows for training and 2,723 rows for testing (`test_data.csv`).
4. Scaled all 16 numeric features using `StandardScaler`, fitted strictly on the training data. KNN, SVM, and Logistic Regression require this because raw features like `Area` (tens of thousands of pixels) and `ShapeFactor` (small decimals) operate on vastly different scales.
5. Saved the fitted `StandardScaler` (`scaler.pkl`) and all 5 trained models (`*.pkl`) in the `model/` directory so the Streamlit app can apply the exact same scaling to uploaded CSV files before predicting.

## c. GitHub repository link
* **GitHub Repository:** https://github.com/harisowmya1094-prog/Dry-Bean-Classification.git/
* **Live Streamlit App:** https://dry-bean-classification-e84cyv9pnlt3qksdeesbbr.streamlit.app/

```text
project-folder/
├── app.py                      # Streamlit app
├── notebook.ipynb              # Trains & saves all 5 models, writes test_data.csv
├── requirements.txt
├── README.md
├── test_data.csv               # Held-out test data (2,723 rows — upload this in the app)
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── svm.pkl
    └── scaler.pkl              # Fitted StandardScaler from training
```

## d. Models used

| Model | Configuration |
| :--- | :--- |
| **Logistic Regression** | `LogisticRegression(max_iter=1000, random_state=42)` |
| **Decision Tree** | `DecisionTreeClassifier(random_state=42)` |
| **KNN** | `KNeighborsClassifier(n_neighbors=5)` |
| **Naive Bayes** | `GaussianNB()` |
| **Support Vector Machine** | `SVC(kernel="rbf", probability=True, random_state=42)` |

### Comparison table (test set, 2,723 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| **Decision Tree** | 0.8920 | 0.9334 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| **KNN** | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| **Naive Bayes** | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| **Support Vector Machine** | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

### Observations
**Ranking by MCC and Accuracy:** Support Vector Machine (0.9059 MCC) > Logistic Regression (0.9042 MCC) > KNN (0.8992 MCC) > Decision Tree (0.8696 MCC) > Naive Bayes (0.7154 MCC)

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Strong runner-up.**<br>- 2nd-best across all metrics: Accuracy 0.9207, AUC 0.9934, F1 0.9209, MCC 0.9042<br>- Standardized geometric features allow linear hyperplanes to separate most bean varieties cleanly<br>- Minor misclassifications occur mostly between morphologically overlapping classes like Sira and Dermason |
| **Decision Tree** | **Moderate baseline, prone to variance.**<br>- Accuracy 0.8920, lowest AUC (0.9334), F1 0.8916, MCC 0.8696<br>- Captures non-linear feature interactions without requiring feature scaling<br>- Axis-aligned splits on continuous morphological measurements generalize less smoothly than margin-based classifiers |
| **KNN** | **Competitive distance-based classifier.**<br>- 3rd-best overall: Accuracy 0.9166, AUC 0.9812, F1 0.9168, MCC 0.8992<br>- Performs well after `StandardScaler` normalizes the 16 feature dimensions<br>- Slightly trails SVM and Logistic Regression in boundary regions where classes overlap |
| **Naive Bayes** | **Weakest overall model.**<br>- Lowest Accuracy (0.7639), Precision (0.7654), Recall (0.7639), F1 (0.7615), and MCC (0.7154)<br>- Retains a respectable AUC (0.9644) for probability ranking, but hard class thresholds suffer<br>- Cause: Strong feature-independence assumption is violated by heavily correlated geometric features (`Area`, `Perimeter`, `ConvexArea`, `EquivDiameter`) |
| **Support Vector Machine** | **Best overall.**<br>- Highest scores across all 6 metrics: Accuracy 0.9221, AUC 0.9937, Precision 0.9224, Recall 0.9221, F1 0.9222, MCC 0.9059<br>- RBF kernel captures non-linear boundaries between similarly shaped beans<br>- Robust to collinearity among derived shape descriptors |
| **Overall winner** | **Support Vector Machine (SVM).**<br>- Leads on every single evaluation metric (Accuracy, AUC, Precision, Recall, F1, and MCC)<br>- Marginally edges out Logistic Regression (0.9059 vs 0.9042 MCC) by modeling non-linear morphological boundaries across the 7 bean classes |

## Streamlit app features
The app provides an interactive interface to test the trained models:
- **Test data:** Upload a custom CSV file or use the built-in `test_data.csv` (2,723 held-out test rows).
- **Model selection:** Choose any of the 5 trained classifiers from the dropdown menu.
- **Model evaluation:** Displays the 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC), confusion matrix, and classification report for the selected model.
- **Compare all models:** Shows a side-by-side comparison table of all 5 models across all 6 metrics and highlights the top-performing model.

The app loads the pre-trained `.pkl` models and `scaler.pkl` directly without retraining. Uploaded CSV files must include the 16 geometric feature columns and the `Class` target column.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying on Streamlit Community Cloud
1. Push this folder to a public GitHub repository.
2. Go to https://streamlit.io/cloud → sign in with GitHub → **New app**.
3. Select the repository, branch `main`, and main file `app.py`.
4. Click **Deploy**.
