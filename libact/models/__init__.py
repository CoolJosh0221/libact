"""
Concrete model classes.
"""
from .logistic_regression import LogisticRegression
from .perceptron import Perceptron
from .svm import SVM
from .sklearn_adapter import SklearnAdapter, SklearnProbaAdapter
# Importable without torch installed; constructing TorchAdapter without
# torch raises an ImportError with installation instructions.
from .torch_adapter import TorchAdapter
