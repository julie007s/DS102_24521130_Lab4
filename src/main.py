from preprocessing import preprocess_data

from decision_tree import run_assignment_1
from random_forest import run_assignment_2
from sklearn_models import run_assignment_3


def main():

    # Load dataset
    X_train, X_test, y_train, y_test = preprocess_data()

    # Assignment 1
    run_assignment_1(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Assignment 2
    run_assignment_2(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Assignment 3
    run_assignment_3(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()