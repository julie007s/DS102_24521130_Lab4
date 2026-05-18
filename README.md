# Wine Quality Classification

This project implements Decision Tree and Random Forest algorithms using NumPy and evaluates them on the Wine Quality dataset.

---

# Assignments

## Assignment 1
- Implement Decision Tree using NumPy
- Train and evaluate using F1-score

## Assignment 2
- Implement Random Forest using NumPy
- Train and evaluate using F1-score

## Assignment 3
- Train and evaluate Decision Tree using scikit-learn
- Train and evaluate Random Forest using scikit-learn

---

# Dataset

Wine Quality Dataset from UCI Machine Learning Repository:

- winequality-red.csv
- winequality-white.csv

The datasets are combined and preprocessed before training.

---

# Preprocessing Steps

- Combine red wine and white wine datasets
- Add wine type feature
- Remove duplicates
- Remove missing values
- Convert quality into classification labels
- Normalize features using StandardScaler
- Split dataset into training and testing sets

---

# Project Structure

```text
Lab 4/
│
├── data/
│   ├── raw/
│   │   ├── winequality-red.csv
│   │   └── winequality-white.csv
│   │
│   └── processed/
│       └── winequality-combined.csv
│
├── results/
│   └── results.md
│
├── src/
│   ├── preprocessing.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── sklearn_models.py
│   └── main.py
│
├── README.md
├── requirements.txt
└── .gitignore