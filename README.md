# LAB4 - SVM Chest X-Ray Classification

## Mô tả bài tập

Bài tập gồm 2 phần:

1. Assignment 1: Cài đặt mô hình Soft-margin SVM bằng NumPy và huấn luyện bằng SGD.
2. Assignment 2: Cài đặt mô hình SVM bằng thư viện sklearn.
3. So sánh hai mô hình bằng Accuracy, Precision, Recall và F1-score.

## Dataset

Dataset sử dụng: Chest X-Ray Images (Pneumonia) trên Kaggle.

Link dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Cấu trúc dataset:

```text
data/chest_xray/train/NORMAL
data/chest_xray/train/PNEUMONIA
data/chest_xray/test/NORMAL
data/chest_xray/test/PNEUMONIA
data/chest_xray/val/NORMAL
data/chest_xray/val/PNEUMONIA
