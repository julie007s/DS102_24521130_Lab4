from sklearn.svm import SVC


def train_sklearn_svm(X_train, y_train):
    """
    Huấn luyện mô hình SVM bằng thư viện sklearn.
    Dùng linear kernel để giống với SVM tuyến tính ở bài 1.
    """

    model = SVC(kernel="linear", C=1.0)

    print("\nĐang train SVM bằng sklearn...")
    model.fit(X_train, y_train)

    return model