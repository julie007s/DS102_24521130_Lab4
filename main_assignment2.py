import json

from src.data_loader import collect, Scaler
from src.svm_sklearn import train_sklearn_svm
from src.evaluate import evaluate_model


def main():
    """
    Assignment 2:
    - Dùng thư viện sklearn để cài đặt SVM
    - Dataset Chest X-Ray Images resize 128 x 128
    - Đánh giá bằng Precision, Recall, F1
    """

    X_train, y_train = collect("train")
    X_test, y_test = collect("test")

    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)

    scaler = Scaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nBÀI 2: SVM SỬ DỤNG THƯ VIỆN SKLEARN")

    model = train_sklearn_svm(X_train, y_train)

    y_pred = model.predict(X_test)

    result = evaluate_model(y_test, y_pred, "Metric from Sklearn SVM")

    with open("result_assignment2.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("\nĐã lưu kết quả vào file result_assignment2.json")


if __name__ == "__main__":
    main()