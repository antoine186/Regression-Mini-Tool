import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split
import random

# mod has to be a string containing the equation of your model. All operations within the string must also be legal within
# Python
# The variable names in inputvar_dt and param_dt must match the corresponding symbols found in the function's internal
# loss function. Furthermore, the ordering of inputvar_dt must correspond to that of the training data's columns.
# The ordering must match between inputvar_dt and param_dt. Finally, inputvar_dt's length must match the number of columns
# found in the training data. The training data must be appended with a first column with all 1 entries
# for the intercept parameter
# obj_type is a string denoting the type of your loss function
# alpha is the learning rate
# Features must be the columns of the training data and the target must be a vector array containing as many instances as
# there are rows in the training data
def simpreg_custom_graddesc(inputvar_dt, param_dt, obj_type, train_type, alpha, train_dt, label_dt):

    if (len(inputvar_dt) != len(param_dt)):

        raise Exception("The number of your input variables does not match the number of the corresponding parameters")

    inputvar_dict = dict()
    param_dict = dict()
    cur_comp_res = np.zeros(len(param_dt))
    contain_err = np.zeros((len(train_dt), len(param_dt) + 1))

    for i in range(len(param_dt)):

        cur_weight = random.randint(0, 10)
        param_dict[param_dt[i]] = cur_weight

    for i in range(len(train_dt)):
        for j in range(len(inputvar_dt)):

            inputvar_dict[inputvar_dt[j]] = train_dt[i, j]

        for j in range(len(param_dt)):

            cur_comp_res[j] = param_dict[param_dt[j]] * inputvar_dict[inputvar_dt[j]]

        cur_comp_pred = np.sum(cur_comp_res)

        cur_err = label_dt[i] - cur_comp_pred
        contain_err[i, 0] = cur_err

        for j in range(len(param_dt)):

            contain_err[i, j + 1] = cur_err * inputvar_dict[inputvar_dt[j]]

        if (train_type == "stochastic"):

            for j in range(len(param_dt)):

                if (j == 1):

                    param_dict[param_dt[j]] = param_dict[param_dt[j]] + (alpha * cur_err)

                else:

                    param_dict[param_dt[j]] = param_dict[param_dt[j]] + (
                                alpha * contain_err[i, j + 1])

    if (train_type == "batch"):

        for j in range(len(param_dt)):

            if (j == 1):

                param_dict[param_dt[j]] = param_dict[param_dt[j]] + (alpha * np.sum(contain_err[:,0]))

            else:

                param_dict[param_dt[j]] = param_dict[param_dt[j]] + (
                        alpha * np.sum(contain_err[:, j + 1]))





