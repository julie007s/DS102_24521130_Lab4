import json
import numpy as np
import matplotlib.pyplot as plt


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_comparison(result_1, result_2):
    """
    Vẽ biểu đồ cột so sánh kết quả bài 1 và bài 2.
    """

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    scratch_scores = [
        result_1["accuracy"],
        result_1["precision"],
        result_1["recall"],
        result_1["f1"]
    ]

    library_scores = [
        result_2["accuracy"],
        result_2["precision"],
        result_2["recall"],
        result_2["f1"]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width / 2, scratch_scores, width, label="SVM tự cài đặt")
    bars2 = plt.bar(x + width / 2, library_scores, width, label="SVM dùng thư viện")

    plt.xticks(x, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Giá trị đánh giá")
    plt.title("So sánh kết quả giữa SVM tự cài đặt và SVM dùng thư viện")
    plt.legend()

    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig("Bang_so_sanh.png", dpi=300)
    plt.show()


def main():
    result_1 = load_result("result_assignment1.json")
    result_2 = load_result("result_assignment2.json")

    plot_comparison(result_1, result_2)


if __name__ == "__main__":
    main()