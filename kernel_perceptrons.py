import matplotlib.pyplot as plt
import math
import numpy as np
import random as rand
import scipy.io


def gaussian_kernel(X, tau):
    X2 = np.sum(np.multiply(X, X), 1)
    K0 = X2 + X2.T - 2 * np.dot(X, X.T)
    K = np.exp(-K0 / tau**2)
    return K


def gaussian_kernelXZ(X, Z, tau):
    X1 = np.sum(np.multiply(X, X), 1)
    X2 = np.sum(np.multiply(Z, Z), 1)
    K0 = X2 + X2.T - 2 * np.dot(X, Z.T)
    K = np.exp(-K0 / tau**2)
    return K


def train_kernel_svm(X, y, lamb, eta, tau, num_iter):
    M = X.shape[0]
    K = gaussian_kernel(X, tau)
    a = np.zeros(M, dtype=np.float64)
    b = 0.0
    l = 0
    for i in range(num_iter):
        t = rand.randint(0, M - 1)
        yt = y[t]
        Kt = np.squeeze(K[:, t])
        at = a[t]
        p = 1.0 - yt * (np.sum(a * Kt) + b)
        lt = lamb * M * np.sum(a * Kt) * at + (p >= 0) * p
        etai = eta / math.sqrt(np.float64(i + 1))
        a = a - etai * (lamb * M * at * Kt - (p >= 0) * yt * Kt)
        b = b + etai * (p >= 0) * yt
        if i == 0:
            l = lt
        l = l * 0.99 + lt * 0.01
    return a, b, l


def train_kernel_svm_minibatch(X, y, lamb, eta, tau, batch_size, num_iter):
    M = X.shape[0]
    K = gaussian_kernel(X, tau)
    a = np.zeros(M, dtype=np.float64)
    b = 0.0
    l = 0
    Kb = np.zeros((M, batch_size))
    yb = np.zeros(batch_size, dtype=np.float64)
    ab = np.zeros(batch_size, dtype=np.float64)
    for i in range(num_iter):
        for b in range(batch_size):
            t = rand.randint(0, M - 1)
            yb[b] = y[t]
            Kb[:, b] = np.squeeze(K[:, t])
            ab[b] = a[t]

        p = 1.0 - yb * (np.dot(a.T, Kb) + b).T
        lt = np.sum(lamb * M * np.dot(a.T, Kb) * ab + (p >= 0) * p) / np.float64(
            batch_size
        )
        etai = eta / (math.sqrt(i + 1) * np.float64(batch_size))
        a = a - etai * (lamb * M * np.dot(Kb, ab) - np.dot(Kb, (p >= 0) * yb))
        b = b + etai * np.sum((p >= 0) * yb)
        if i == 0:
            l = lt
        l = l * 0.99 + lt * 0.01
    return a, b, l


def train_kernel_perceptron(X, y, tau, num_iter):
    M = X.shape[0]
    K = gaussian_kernel(X, tau)
    a = np.zeros(M, dtype=np.float64)
    for i in range(num_iter):
        t = rand.randint(0, M - 1)
        yt = y[t]
        Kt = np.squeeze(K[:, t])
        at = a[t]
        p = np.sign(np.sum(a * Kt))
        if p != yt:
            a[t] = a[t] + yt
    return a


def plot(Xtrain, Xtest, ytest, a, b=0, fig_name="banana.pdf"):
    plot_step = 0.02
    x_min, x_max = Xtrain[:, 0].min(), Xtrain[:, 0].max()
    y_min, y_max = Xtrain[:, 1].min(), Xtrain[:, 1].max()
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    Z = np.c_[xx.ravel(), yy.ravel()]
    K = gaussian_kernelXZ(Xtrain, Z, tau)
    scores = np.dot(a.T, K)
    scores = scores.reshape(xx.shape)

    pos = np.where(ytest == 1)
    neg = np.where(ytest == -1)
    fig = plt.figure()
    plt.scatter(Xtest[pos, 0], Xtest[pos, 1], marker="o", c="b")
    plt.scatter(Xtest[neg, 0], Xtest[neg, 1], marker="o", c="r")
    plt.contour(xx, yy, scores, colors="black", linewidths=2)
    plt.tick_params(
        axis="x",
        which="both",
        bottom="off",
        top="off",
        labelbottom="off",
    )
    plt.tick_params(
        axis="y",
        which="both",
        left="off",
        right="off",
        labelleft="off",
    )
    plt.show()
    fig.savefig(fig_name, bbox_inches="tight")


if __name__ == "__main__":

    banana = scipy.io.loadmat("banana.mat")
    Xtrain = banana["xTest"].T
    Xtest = banana["xTrain"].T
    ytrain = np.squeeze(banana["yTest"])
    ytest = np.squeeze(banana["yTrain"])

    pos = np.where(ytrain == 1)
    neg = np.where(ytrain == -1)

    tau = 1.0
    num_iter = 5000
    b = 0
    ap = train_kernel_perceptron(Xtrain, ytrain, tau, num_iter)
    plot(Xtrain, Xtest, ytest, ap)

    asvm, bsvm, loss_svm = train_kernel_svm(Xtrain, ytrain, 1e-8, 1.0, tau, num_iter)
    plot(Xtrain, Xtest, ytest, asvm, bsvm)
    amsvm, bmsvm, loss_msvm = train_kernel_svm_minibatch(
        Xtrain, ytrain, 1e-8, 1.0, tau, 1, num_iter
    )
    plot(Xtrain, Xtest, ytest, amsvm, bmsvm)
