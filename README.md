# Wine Quality Classification

This project implements Decision Tree and Random Forest algorithms using NumPy and evaluates them on the Wine Quality dataset using Weighted F1-score.

---

# Objectives

The objectives of this project are:

- Implement Decision Tree from scratch using NumPy
- Implement Random Forest from scratch using NumPy
- Train and evaluate machine learning models
- Compare custom implementations with scikit-learn models

---

# Dataset

Dataset: Wine Quality Dataset  
Source: UCI Machine Learning Repository

The project uses:

- winequality-red.csv
- winequality-white.csv

The two datasets are combined into a single dataset before preprocessing and training.

---

# Data Preprocessing

The preprocessing pipeline includes:

1. Loading red wine and white wine datasets
2. Adding wine type feature (`is_red`)
3. Combining datasets
4. Removing missing values
5. Removing duplicate samples
6. Converting quality scores into classification labels
7. Standardizing numerical features
8. Splitting data into training and testing sets

---

# Theory

## Decision Tree

Decision Tree is a supervised learning algorithm used for classification and regression tasks.

The model recursively splits the dataset into smaller subsets based on the feature that provides the highest Information Gain.

### Entropy

Entropy measures the impurity of a dataset:

```math
H(S) = -\sum p_i \log_2(p_i)
````

### Information Gain

Information Gain measures the reduction in entropy after a split:

```math
IG(S, A) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)
```

---

## Random Forest

Random Forest is an ensemble learning method that combines multiple Decision Trees.

Main ideas of Random Forest:

* Bootstrap Sampling
* Random Feature Selection
* Majority Voting

Advantages:

* Reduces overfitting
* Improves generalization
* More stable than a single Decision Tree

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
```

---

# Experimental Results

| Model         | Implementation | Weighted F1-score |
| ------------- | -------------- | ----------------- |
| Decision Tree | NumPy          | 0.6824            |
| Random Forest | NumPy          | 0.7685            |
| Decision Tree | scikit-learn   | 0.7210            |
| Random Forest | scikit-learn   | 0.7603            |

---

# Discussion

## Decision Tree

The custom NumPy Decision Tree achieved a Weighted F1-score of 0.6824.

Although the performance is lower than the scikit-learn implementation, the result demonstrates that the algorithm was implemented successfully.

The performance difference is mainly caused by:

* Simpler splitting strategy
* Lack of pruning
* Fewer optimizations compared to scikit-learn

---

## Random Forest

The custom Random Forest achieved the best performance with a Weighted F1-score of 0.7685.

The ensemble learning approach improved performance significantly compared to a single Decision Tree.

Random Forest reduces overfitting by:

* Training multiple trees
* Using bootstrap sampling
* Randomly selecting features at each split

---

# Conclusion

This project successfully implemented:

* Decision Tree using NumPy
* Random Forest using NumPy
* Model evaluation using Weighted F1-score

The experiments show that Random Forest performs better than a single Decision Tree.

The custom NumPy implementations achieved results close to the scikit-learn models.

---

# How to Run

## 1. Create virtual environment

```bash id="tsjlwm"
python -m venv venv
```

## 2. Activate virtual environment

### Windows PowerShell

```bash id="jlwmg9"
venv\Scripts\Activate.ps1
```

### Git Bash

```bash id="jlwmf8"
source venv/Scripts/activate
```

## 3. Install dependencies

```bash id="jlwmj7"
pip install -r requirements.txt
```

## 4. Run the project

```bash id="jlwmk6"
python src/main.py
```

---

# Author

Ngân Nguyễn

```

