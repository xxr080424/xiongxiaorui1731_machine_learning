from ucimlrepo import fetch_ucirepo
import numpy as np


def sigmoid(z):
    """Sigmoid activation function: maps any real number to a probability in (0, 1)."""
    return 1 / (1 + np.exp(-z))


def regularized_logistic_regression(x, y, learning_rate=0.1, n_iterations=5000, lam=0.1):
    """
    Logistic regression with L2 regularization (gradient descent implementation).

    Model:     p = sigmoid(x @ w + b)
    Loss:      J = -(1/n) * Σ [ y*log(p) + (1-y)*log(1-p) ] + (lam/(2n)) * ||w||^2
    Gradients: dw = (1/n) * x.T @ (p - y) + (lam/n) * w
               db = (1/n) * Σ (p - y)

    :param x:             feature matrix, shape (n, d), already standardized
    :param y:             labels, shape (n,), taking values 0 or 1
    :param learning_rate: learning rate
    :param n_iterations:  number of iterations
    :param lam:           L2 regularization coefficient (larger => simpler model, less likely to overfit)
    :return: w, b, and the list of loss values at each iteration
    """
    n, d = x.shape
    w = np.zeros(d)          # one weight per feature
    b = 0.0
    losses = []

    for _ in range(n_iterations):
        p = sigmoid(x @ w + b)                     # predicted probability
        # regularized log-likelihood loss (cross-entropy + L2 penalty)
        eps = 1e-8                                 # prevent log(0)
        loss = -(1 / n) * np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss += (lam / (2 * n)) * np.sum(w ** 2)   # L2 regularization term
        losses.append(loss)

        dw = (1 / n) * (x.T @ (p - y)) + (lam / n) * w   # note the regularization only applies to w, not b
        db = (1 / n) * np.sum(p - y)
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, losses


def accuracy(x, y, w, b):
    """Compute classification accuracy: predicted probability >= 0.5 is classified as 1, otherwise 0."""
    p = sigmoid(x @ w + b)
    y_pred = (p >= 0.5).astype(int)
    return np.mean(y_pred == y)


if __name__ == "__main__":
    # 1) Load the binary classification dataset: Breast Cancer Wisconsin (Diagnostic) (id=17)
    #    30 numerical features, label Diagnosis is 'M' (malignant) / 'B' (benign)
    cancer = fetch_ucirepo(id=17)

    # 2) Convert to numpy arrays and standardize the features
    x = cancer.data.features.to_numpy(dtype=float)                 # shape: (n, 30)
    y = (cancer.data.targets.to_numpy().ravel() == 'M').astype(int)  # 'M' -> 1, 'B' -> 0
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    # 3) Simple train/test split (80% train, 20% test); fit on train, evaluate generalization on test
    n = len(x)
    idx = np.random.permutation(n)
    n_train = int(0.8 * n)
    x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]
    x_test, y_test = x[idx[n_train:]], y[idx[n_train:]]

    # 4) Train two models for comparison:
    #    lam=0   -- no regularization (may overfit, large train/test accuracy gap)
    #    lam=0.1 -- L2 regularization (limits weight magnitude, usually generalizes better)
    for lam in (0.0, 0.1):
        w, b, losses = regularized_logistic_regression(
            x_train, y_train, learning_rate=0.1, n_iterations=5000, lam=lam
        )
        acc_train = accuracy(x_train, y_train, w, b)
        acc_test = accuracy(x_test, y_test, w, b)
        print(f"正则化系数 λ = {lam}")
        print(f"  训练集准确率: {acc_train:.4f}")
        print(f" 测试集准确率:     {acc_test:.4f}")
        print(f" 最终损失值:  {losses[-1]:.6f}")
        print()

