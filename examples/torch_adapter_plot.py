#!/usr/bin/env python3
"""
This example shows how to plug a PyTorch neural network into libact's
active-learning loop with TorchAdapter, comparing UncertaintySampling
against RandomSampling on the scikit-learn digits dataset.

PyTorch is an optional dependency of libact; install it with
`pip install libact[torch]` (or `pip install torch`) before running.
"""

import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch.nn as nn

# libact classes
from libact.base.dataset import Dataset
from libact.models import TorchAdapter
from libact.query_strategies import RandomSampling, UncertaintySampling
from libact.labelers import IdealLabeler


def make_net():
    """A small MLP for the 8x8 digits (64 features, 10 classes)."""
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def make_model(seed):
    return TorchAdapter(make_net, classes=list(range(10)), n_epochs=20,
                        batch_size=64, random_state=seed)


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        lb = lbr.label(trn_ds.data[ask_id][0])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test(test_size, n_labeled):
    digits = load_digits()
    X = digits.data / 16.0  # scale pixel values to [0, 1]
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1126, stratify=y)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, fully_labeled_trn_ds


def main():
    test_size = 0.33    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 30      # number of samples that are initially labeled
    quota = 50          # number of samples to query

    # Load dataset
    trn_ds, tst_ds, fully_labeled_trn_ds = \
        split_train_test(test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    # Comparing UncertaintySampling strategy with RandomSampling.
    # The model in both loops is a PyTorch MLP wrapped in TorchAdapter.
    qs = UncertaintySampling(trn_ds, method='lc', model=make_model(1126))
    model = make_model(1126)
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)

    qs2 = RandomSampling(trn_ds2, random_state=1126)
    model = make_model(1126)
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b', label='qs Ein')
    plt.plot(query_num, E_in_2, 'r', label='random Ein')
    plt.plot(query_num, E_out_1, 'g', label='qs Eout')
    plt.plot(query_num, E_out_2, 'k', label='random Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result (PyTorch MLP via TorchAdapter)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
