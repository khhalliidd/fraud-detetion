# 💳 Credit Card Fraud Detection

This project uses supervised machine learning techniques to detect fraudulent credit card transactions. It includes thorough data exploration, feature engineering, model evaluation, and comparison of multiple algorithms. The dataset is highly imbalanced, making fraud detection a challenging but rewarding task.

---

## 📁 Project Overview

The main objectives of this project are:

- Detect fraudulent credit card transactions.
- Handle class imbalance effectively.
- Compare multiple classification models using evaluation metrics.
- Apply feature selection using ANOVA for performance improvement.

The entire analysis is performed in the Jupyter Notebook: `fraud_detection.ipynb`.

---

## 🧠 Dataset

The dataset used is `creditcard.csv`, which contains transactions made by European cardholders in September 2013. The features were transformed using PCA for anonymity.

- **Rows:** 284,807
- **Columns:** 31
- **Features:**
  - `Time`: Time elapsed between transactions.
  - `V1` to `V28`: PCA-transformed features.
  - `Amount`: Transaction amount.
  - `Class`: Target variable (0 = Non-Fraud, 1 = Fraud)

---

## 📊 Models and Results

The performance of five machine learning models is compared before and after feature selection using ANOVA.

### 🔹 Original Model Results

| # | Algorithm                  | Cross-Validation | ROC AUC  | F1 Score (Fraud) |
|---|----------------------------|------------------|----------|------------------|
| 1 | Logistic Regression        | 98.01%           | 92.35%   | 91%              |
| 2 | Support Vector Classifier  | 97.94%           | 92.10%   | 91%              |
| 3 | Decision Tree Classifier   | 96.67%           | 91.36%   | 90%              |
| 4 | Random Forest Classifier   | 97.84%           | 91.71%   | 91%              |
| 5 | K-Nearest Neighbors        | 99.34%           | 97.63%   | 97%              |

### 🔹 ANOVA Feature Selection Results

| # | Algorithm                  | Cross-Validation | ROC AUC  | F1 Score (Fraud) |
|---|----------------------------|------------------|----------|------------------|
| 1 | Logistic Regression        | 98.45%           | 94.69%   | 94%              |
| 2 | Support Vector Classifier  | 98.32%           | 94.40%   | 94%              |
| 3 | Decision Tree Classifier   | 97.13%           | 93.69%   | 93%              |
| 4 | Random Forest Classifier   | 98.20%           | 94.06%   | 94%              |
| 5 | K-Nearest Neighbors        | 99.54%           | 98.47%   | 97%              |

---

## ⚙️ Dependencies

Make sure the following Python libraries are installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

### Install with pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

Download the dataset creditcard.csv and place it in the root directory.

Open fraud_detection.ipynb using Jupyter Notebook or JupyterLab.

Run all cells sequentially to:

Load and explore the data

Train multiple models

Evaluate performance

Apply ANOVA feature selection
