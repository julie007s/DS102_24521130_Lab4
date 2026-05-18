import numpy as np
from sklearn.metrics import f1_score

from decision_tree import DecisionTreeClassifier


# ====================================
# Random Forest Classifier


class RandomForestClassifier:

    def __init__(
        self,
        n_trees=10,
        max_depth=10,
        min_samples_split=2,
        n_features=None
    ):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self.trees = []

    # ====================================

    def fit(self, X, y):

        self.trees = []

        n_samples, n_features_total = X.shape

        # Number of random features
        if self.n_features is None:

            self.n_features = int(
                np.sqrt(n_features_total)
            )

        # Train multiple trees
        for _ in range(self.n_trees):

            # Bootstrap sampling
            sample_indices = np.random.choice(
                n_samples,
                n_samples,
                replace=True
            )

            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Create Decision Tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )

            # Train tree
            tree.fit(X_sample, y_sample)

            # Save tree
            self.trees.append(tree)

    # ====================================

    def predict(self, X):

        # Predictions from all trees
        tree_predictions = np.array(
            [
                tree.predict(X)
                for tree in self.trees
            ]
        )

        # Shape:
        # (n_trees, n_samples)
        # -> transpose
        tree_predictions = np.swapaxes(
            tree_predictions,
            0,
            1
        )

        final_predictions = []

        # Majority voting
        for predictions in tree_predictions:

            predictions = np.array(
                predictions,
                dtype=int
            )

            voted_prediction = np.argmax(
                np.bincount(predictions)
            )

            final_predictions.append(
                voted_prediction
            )

        return np.array(final_predictions)


# ====================================
# Assignment 2 Runner


def run_assignment_2(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n ASSIGNMENT 2 ")

    model = RandomForestClassifier(
        n_trees=15,
        max_depth=12,
        min_samples_split=5
    )

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"Random Forest F1-score: {f1:.4f}")

    print("\nAssignment 2 completed!\n")

    return f1