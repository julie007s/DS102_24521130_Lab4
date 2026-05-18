import pandas as pd
import numpy as np


class StandardScaler:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):

        X = np.array(X, dtype=float)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        # Avoid division by zero
        self.std[self.std == 0] = 1

        return (X - self.mean) / self.std

    def transform(self, X):

        X = np.array(X, dtype=float)

        return (X - self.mean) / self.std


def train_test_split_custom(X, y, test_size=0.2, random_state=42):

    np.random.seed(random_state)

    n_samples = len(X)

    indices = np.arange(n_samples)

    np.random.shuffle(indices)

    test_count = int(n_samples * test_size)

    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


def preprocess_data():

    print("Loading datasets...")

    # Load datasets
    red_df = pd.read_csv(
        "data/raw/winequality-red.csv",
        sep=";"
    )

    white_df = pd.read_csv(
        "data/raw/winequality-white.csv",
        sep=";"
    )

    # Add wine type column
    red_df["type"] = 0
    white_df["type"] = 1

    print("Combining datasets...")

    # Combine datasets
    df = pd.concat(
        [red_df, white_df],
        ignore_index=True
    )

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    print("Converting labels...")

    # Binary classification
    # Good wine = 1
    # Bad wine = 0
    df["quality"] = (
        df["quality"] >= 6
    ).astype(int)

    # Save processed dataset
    df.to_csv(
        "data/processed/winequality-combined.csv",
        index=False
    )

    print("Splitting dataset...")

    # Features and labels
    X = df.drop(columns=["quality"])
    y = df["quality"]

    # Split train/test
    X_train, X_test, y_train, y_test = (
        train_test_split_custom(
            X,
            y,
            test_size=0.2
        )
    )

    print("Normalizing data...")

    # Normalize data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to NumPy arrays
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print("\nPreprocessing completed!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    preprocess_data()