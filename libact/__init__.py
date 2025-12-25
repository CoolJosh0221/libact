"""
libact
======

Pool-based Active Learning in Python.

libact is a Python package designed to make active learning easier for
real-world users. The package implements several popular active learning
strategies and features the active-learning-by-learning meta-algorithm.

Available subpackages
---------------------
base
    Defines interface and dataset object.
labelers
    Interface for labeling data.
models
    Learning models.
query_strategies
    Implementation of active learning algorithms.

Quick Start
-----------
>>> from libact.base.dataset import Dataset
>>> from libact.query_strategies import UncertaintySampling
>>> # Check what optional features are available
>>> from libact import check_optional_features
>>> check_optional_features()

"""

__version__ = "0.1.6"
__all__ = ["base", "labelers", "models", "query_strategies",
           "check_optional_features", "get_available_features"]


def check_optional_features(verbose=True):
    """
    Check and report the availability of optional features.

    Some libact features (HintSVM, VarianceReduction) require BLAS/LAPACK
    libraries to be installed. This function checks which features are
    available in your installation.

    Parameters
    ----------
    verbose : bool, default=True
        If True, print the status of each optional feature.

    Returns
    -------
    dict
        Dictionary with feature names as keys and boolean availability as values.

    Examples
    --------
    >>> import libact
    >>> features = libact.check_optional_features()
    libact Optional Features Status
    ========================================
      variance_reduction: ✓ Available
      hintsvm: ✓ Available
    ========================================
    """
    from .query_strategies import check_optional_features as _check
    return _check(verbose=verbose)


def get_available_features():
    """
    Return a dictionary of optional features and their availability.

    Returns
    -------
    dict
        Dictionary with feature names as keys and boolean availability as values.
        Example: {'variance_reduction': True, 'hintsvm': False}

    Examples
    --------
    >>> import libact
    >>> libact.get_available_features()
    {'variance_reduction': True, 'hintsvm': True}
    """
    from .query_strategies import get_available_features as _get
    return _get()
