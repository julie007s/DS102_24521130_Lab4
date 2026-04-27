import os
import random
import cv2 as cv
import numpy as np
from tqdm import tqdm

IMG_SIZE = 128
NORMAL_LABEL = -1
PNEUMONIA_LABEL = 1


def collect(split: str = "train", base_dir: str = "data/chest_xray"):
    """
    Đọc ảnh từ thư mục train/test/val.
    Quy trình:
    - đọc ảnh
    - chuyển sang ảnh xám
    - resize về 128 x 128
    - flatten thành vector 1 chiều
    - gán nhãn: NORMAL = -1, PNEUMONIA = 1
    """

    images = []
    labels = []

    class_map = {
        "NORMAL": NORMAL_LABEL,
        "PNEUMONIA": PNEUMONIA_LABEL
    }

    for class_name, label in class_map.items():
        class_dir = os.path.join(base_dir, split, class_name)

        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {class_dir}")

        for img_file in tqdm(os.listdir(class_dir), desc=f"Loading {split}/{class_name}"):
            img_path = os.path.join(class_dir, img_file)

            img = cv.imread(img_path)

            if img is None:
                print("Cannot read:", img_path)
                continue

            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv.INTER_LINEAR_EXACT)
            img = img.reshape(-1)

            images.append(img)
            labels.append(label)

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.array(labels, dtype=np.int32)

    data = list(zip(X, y))
    random.shuffle(data)

    X, y = zip(*data)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y


class Scaler:
    """
    Chuẩn hóa dữ liệu bằng mean và std của tập train.
    Chỉ fit trên train, sau đó dùng transform cho test.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        X_scaled = (X - self.mean) / (self.std + 1e-8)

        return X_scaled

    def transform(self, X: np.ndarray):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        X_scaled = (X - self.mean) / (self.std + 1e-8)

        return X_scaled