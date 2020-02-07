from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def simply_toy_reg(gauss_noise, test_size, reg_samples = 100, nb_feats = 1):

    X, y = make_regression(n_samples = reg_samples, n_features = nb_feats, noise = gauss_noise)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=None)

    lin_mod = linear_model.LinearRegression()
    lin_mod.fit(X_train, y_train)

    plt.scatter(X_train, y_train, color="black")
    plt.scatter(X_test, y_test, color="red")
    plt.scatter(X_test, lin_mod.predict(X_test), color="blue")

    plt.title('Simple Regression Data within 2D Feature Space')
    plt.xlabel("X Input", fontsize=8)
    plt.ylabel("Y Output", fontsize=8)
    plt.axis("tight")

    print("Mean squared error: %.2f" % np.mean((lin_mod.predict(X_test) - y_test) ** 2))
    print("Variance score: % .2f" % lin_mod.score(X_test, y_test))