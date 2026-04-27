import numpy as np


class SVM:
    def __init__(self, C: float = 1.0, lr: float = 0.0001, epoch: int = 5):
        self.C = C
        self.lr = lr
        self.epoch = epoch
        self.w = None
        self.b = None
        self.losses = []

    def predict(self, X: np.ndarray):
        return X @ self.w + self.b

    def predict_class(self, X: np.ndarray):
        y_hat = self.predict(X)
        return np.where(y_hat >= 0, 1, -1).flatten()

    def loss_fn(self, X: np.ndarray, y: np.ndarray):
        y = y.reshape(-1, 1)
        y_hat = self.predict(X)
        margins = 1 - y * y_hat
        return 0.5 * np.sum(self.w ** 2) + self.C * np.maximum(0, margins).mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        N, dim = X.shape
        self.w = np.zeros((dim, 1))
        self.b = 0.0
        y = y.reshape(-1, 1)

        for epoch in range(self.epoch):
            indices = np.random.permutation(N)
            X = X[indices]
            y = y[indices]

            for i in range(N):
                x_i = X[i].reshape(1, -1)
                y_i = y[i][0]

                y_hat_i = self.predict(x_i)[0, 0]

                if y_i * y_hat_i >= 1:
                    dW = self.w
                    db = 0
                else:
                    dW = self.w - self.C * y_i * x_i.T
                    db = -self.C * y_i

                self.w = self.w - self.lr * dW
                self.b = self.b - self.lr * db

            loss = self.loss_fn(X, y)
            self.losses.append(loss)

            print(f"Epoch {epoch + 1}/{self.epoch} - Loss: {loss:.4f}")