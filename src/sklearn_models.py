from sklearn.tree import DecisionTreeClassifier as SklearnTree
from sklearn.ensemble import RandomForestClassifier as SklearnForest
from sklearn.metrics import f1_score


def run_assignment_3(X_train, X_test, y_train, y_test):

    print("\n========== ASSIGNMENT 3 ==========")

    # ====================================
    # Decision Tree using scikit-learn
    # ====================================

    print("\nTraining Decision Tree...")

    dt_model = SklearnTree(
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )

    dt_model.fit(X_train, y_train)

    dt_predictions = dt_model.predict(X_test)

    dt_f1 = f1_score(
        y_test,
        dt_predictions,
        average="weighted"
    )

    print(f"Decision Tree F1-score: {dt_f1:.4f}")

    # ====================================
    # Random Forest using scikit-learn
    # ====================================

    print("\nTraining Random Forest...")

    rf_model = SklearnForest(
        n_estimators=15,
        max_depth=12,
        min_samples_split=5,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_predictions = rf_model.predict(X_test)

    rf_f1 = f1_score(
        y_test,
        rf_predictions,
        average="weighted"
    )

    print(f"Random Forest F1-score: {rf_f1:.4f}")

    print("\nAssignment 3 completed!\n")

    return dt_f1, rf_f1