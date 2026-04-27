import json

from src.data_loader import collect, Scaler
from src.svm_from_scratch import SVM
from src.evaluate import evaluate_model


def main():
    """
    Assignment 1:
    - Dùng NumPy cài đặt soft-margin SVM
    - Train bằng SGD
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

    model = SVM(C=1.0, lr=0.0001, epoch=5)

    print("\nBÀI 1: SVM TỰ CÀI ĐẶT BẰNG NUMPY")
    model.fit(X_train, y_train)

    y_pred = model.predict_class(X_test)

    result = evaluate_model(y_test, y_pred, "Metric from Scratch SVM")

    with open("result_assignment1.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print("\nĐã lưu kết quả vào file result_assignment1.json")


if __name__ == "__main__":
    main()