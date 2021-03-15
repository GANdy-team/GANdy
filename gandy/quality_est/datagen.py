"""
This module generates noisy data.

Data is either generated analytically or from the QM9 dataset.
"""
# standard imports
import numpy as np
import pandas as pd

# deepchem data imports
from deepchem.molnet import load_qm9


def generate_analytical_data(to_csv=True):
    """
    Generate noisy analytical data.

    This function generates random x1 and x2 as data features,
    creates and analytical target value using the f function,
    and then adds noise to that value using g.
    """
    x1 = np.random.uniform(0, 10, 10000)
    x2 = np.random.uniform(0, 10, 10000)
    mu = 0
    sigma = (x1 + x2) / 10

    def f(x1, x2):
        f_data = 2*x1 + x2
        return f_data

    def g(x1, x2):
        g_data = np.random.normal(mu, np.abs(sigma), 10000)
        return g_data

    noise = g(x1, x2)
    y = f(x1, x2) + noise

    gen_data = pd.DataFrame({'X1': x1, 'X2': x2, 'Y': y})

    if to_csv:
        gen_data.to_csv("analytical_data.csv", index=False, sep=',')
        # read in using gen_data = pd.read_csv("analytical_data.csv")
    return gen_data, noise


def generate_qm9_noise_data(x1, x2, y, to_csv=True):
    """
    Generate noisy QM9 data.

    This function takes in x1 and x2, which correspond to
    the data columns in QM9 to use as data (in [1,12]).

    The y is the column to use as the target, also in [1,12].
    """
    # load data
    qm9_tasks, datasets, transformers = load_qm9()
    train_dataset, valid_dataset, test_dataset = datasets

    c1 = qm9_tasks[x1 - 1]
    c2 = qm9_tasks[x2 - 1]
    c3 = qm9_tasks[y - 1]

    # extrct the 'y'values
    Y = test_dataset.y
    YT = Y.T

    X1 = YT[x1 - 1]
    X2 = YT[x2 - 1]
    Y_a = YT[y - 1]

    x1 = X1.tolist()
    x2 = X2.tolist()
    y_l = Y_a.tolist()
    length = len(Y_a)


    # add noise to n numbers of y
    Noise = []
    for i in range(length):
        mu = 0
        sigma = (x1[i] + x2[i]) / 2
        noise = np.random.normal(mu, np.abs(sigma), length)
        g = noise.tolist()
        Noise.append(g[i])
        y_l[i] += g[i]

    # save to_csv:
    gen_data = pd.DataFrame({f'x1_{c1}': X1, f'x2_{c2}': X2, f'y_{c3}': y_l})

    if to_csv:
        gen_data.to_csv("qm9_noise_data.csv", index=False, sep=',')
        # read in using gen_data = pd.read_csv('qm9_noise_data.csv')

    return gen_data, Noise
