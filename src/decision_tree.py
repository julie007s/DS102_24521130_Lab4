import numpy as np
from sklearn.metrics import f1_score


# ====================================
# Entropy Function


def entropy(y):

    y = np.array(y, dtype=int)

    if len(y) == 0:
        return 0

    class_counts = np.bincount(y)

    probabilities = class_counts / len(y)

    return -np.sum(
        [
            p * np.log2(p)
            for p in probabilities
            if p > 0
        ]
    )


# ====================================
# Tree Node


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):

        return self.value is not None


# ====================================
# Decision Tree Classifier


class DecisionTreeClassifier:

    def __init__(
        self,
        min_samples_split=2,
        max_depth=10,
        n_features=None
    ):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    # ====================================

    def fit(self, X, y):

        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(
                X.shape[1],
                self.n_features
            )

        self.root = self._grow_tree(X, y)

    # ====================================

    def _grow_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):

            leaf_value = self._most_common_label(y)

            return Node(value=leaf_value)

        # Random feature selection
        feature_indices = np.random.choice(
            n_features,
            self.n_features,
            replace=False
        )

        # Find best split
        best_feature, best_threshold = self._best_split(
            X,
            y,
            feature_indices
        )

        # If no split found
        if best_feature is None:

            leaf_value = self._most_common_label(y)

            return Node(value=leaf_value)

        # Split dataset
        left_indices, right_indices = self._split(
            X[:, best_feature],
            best_threshold
        )

        # Build subtrees recursively
        left_subtree = self._grow_tree(
            X[left_indices, :],
            y[left_indices],
            depth + 1
        )

        right_subtree = self._grow_tree(
            X[right_indices, :],
            y[right_indices],
            depth + 1
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )

    # ====================================

    def _best_split(self, X, y, feature_indices):

        best_gain = -1

        split_feature = None
        split_threshold = None

        for feature_index in feature_indices:

            X_column = X[:, feature_index]

            thresholds = np.unique(X_column)

            for threshold in thresholds:

                gain = self._information_gain(
                    y,
                    X_column,
                    threshold
                )

                if gain > best_gain:

                    best_gain = gain

                    split_feature = feature_index
                    split_threshold = threshold

        return split_feature, split_threshold

    # ====================================

    def _information_gain(
        self,
        y,
        X_column,
        threshold
    ):

        parent_entropy = entropy(y)

        left_indices, right_indices = self._split(
            X_column,
            threshold
        )

        if (
            len(left_indices) == 0
            or len(right_indices) == 0
        ):
            return 0

        n = len(y)

        n_left = len(left_indices)
        n_right = len(right_indices)

        left_entropy = entropy(y[left_indices])

        right_entropy = entropy(y[right_indices])

        child_entropy = (
            (n_left / n) * left_entropy
            +
            (n_right / n) * right_entropy
        )

        information_gain = (
            parent_entropy - child_entropy
        )

        return information_gain

    # ====================================

    def _split(self, X_column, threshold):

        left_indices = np.argwhere(
            X_column <= threshold
        ).flatten()

        right_indices = np.argwhere(
            X_column > threshold
        ).flatten()

        return left_indices, right_indices

    # ====================================

    def _most_common_label(self, y):

        y = np.array(y, dtype=int)

        return np.argmax(np.bincount(y))

    # ====================================

    def predict(self, X):

        predictions = [
            self._traverse_tree(x, self.root)
            for x in X
        ]

        return np.array(predictions)

    # ====================================

    def _traverse_tree(self, x, node):

        if node.is_leaf_node():

            return node.value

        if x[node.feature] <= node.threshold:

            return self._traverse_tree(
                x,
                node.left
            )

        return self._traverse_tree(
            x,
            node.right
        )


# ====================================
# Assignment 1 Runner


def run_assignment_1(
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n ASSIGNMENT 1 ")

    model = DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=5
    )

    print("\nTraining Decision Tree...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print(f"Decision Tree F1-score: {f1:.4f}")

    print("\nAssignment 1 completed!\n")

    return f1