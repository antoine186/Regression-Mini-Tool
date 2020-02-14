# Regression via Custom Gradient Descent Tool

## Overview

This is a function implementing a regression model obtained via the application of a custom-built gradient descent method.

(Note) This function requires Python 3.6 or higher. This tool is released with the required dependencies found in the venv folder.

## How to Use
### Univariate Regression Gradient Descent

Let us first import the required dependencies:

```
from SimpleRegMod import simply_toy_reg
from sklearn.datasets import make_regression
from CustomGradDescReg import simpreg_custom_graddesc
import matplotlib.pyplot as plt
import numpy as np
```

Let us generate some random data upon which we will fit our regression model and perform gradient descent:
(Note): the concatenation of an all ones column at the front of the generated data is meant to maintain the contribution of the intercept as 1.

```
X, y = make_regression(n_samples = 100, n_features = 1, noise = 3)
X_head = np.ones((100,1))
new_X = np.concatenate((X_head, X), axis=1)
```

We want to learn a univariate regression:

```
inputvar = np.array(["x0", "x1"])
params = np.array(["w0", "w1"])
```

res_stoch, err_stoch = simpreg_custom_graddesc(inputvar, params, train_type="stochastic",
                              alpha=0.05, train_dt=new_X, label_dt=y, iter_nb=3)

x_vals = np.arange(len(err_stoch))

plt.plot(x_vals, err_stoch)

plt.title('Error Rate Over the Number of Predictions')
plt.xlabel("Nb. of Predictions", fontsize=8)
plt.ylabel("Error Value", fontsize=8)
plt.axis("tight")

res_batch, err_batch = simpreg_custom_graddesc(inputvar, params, train_type="batch",
                              alpha=0.005, train_dt=new_X, label_dt=y, iter_nb=10)

x_vals = np.arange(len(err_batch))

plt.plot(x_vals, err_batch)

plt.title('Error Rate Over the Number of Predictions')
plt.xlabel("Nb. of Predictions", fontsize=8)
plt.ylabel("Error Value", fontsize=8)
plt.axis("tight")
