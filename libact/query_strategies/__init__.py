"""
Concrete query strategy classes.

Available Query Strategies
--------------------------
Core strategies (always available):
- ActiveLearningByLearning
- DWUS (Density-Weighted Uncertainty Sampling)
- QUIRE
- QueryByCommittee
- RandomSampling
- UncertaintySampling
- DensityWeightedMeta

Optional strategies (require BLAS/LAPACK):
- HintSVM
- VarianceReduction
"""
from __future__ import absolute_import

import os
import sys
import warnings

ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'

# Core strategies (always available)
from .active_learning_by_learning import ActiveLearningByLearning
from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
from .density_weighted_uncertainty_sampling import DWUS
from .density_weighted_meta import DensityWeightedMeta

# Track available optional features
_OPTIONAL_FEATURES = {
    'variance_reduction': False,
    'hintsvm': False,
}


def _format_install_hint():
    """Generate platform-specific installation hints."""
    if sys.platform.startswith('linux'):
        return (
            "On Debian/Ubuntu: sudo apt-get install libopenblas-dev liblapacke-dev\n"
            "On Fedora/RHEL: sudo dnf install openblas-devel lapack-devel\n"
            "On Arch Linux: sudo pacman -S openblas lapacke"
        )
    elif sys.platform == 'darwin':
        return (
            "On macOS: brew install openblas\n"
            "Then reinstall libact: pip install --force-reinstall libact"
        )
    elif sys.platform == 'win32':
        return (
            "On Windows, BLAS/LAPACK support requires additional setup.\n"
            "Options:\n"
            "  1. Use conda: conda install -c conda-forge openblas\n"
            "  2. Install Intel MKL: pip install mkl\n"
            "Then reinstall libact: pip install --force-reinstall libact"
        )
    else:
        return "Please install BLAS/LAPACK development libraries for your platform."


# Don't import C extensions when on ReadTheDocs server
if not ON_RTD:
    # Try to import VarianceReduction
    try:
        from ._variance_reduction import estVar
        from .variance_reduction import VarianceReduction
        _OPTIONAL_FEATURES['variance_reduction'] = True
    except ImportError:
        # Create a placeholder class that raises an informative error
        class VarianceReduction:
            """VarianceReduction query strategy (not available - requires BLAS/LAPACK)."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "VarianceReduction is not available because the C extension was not built.\n"
                    "This usually means BLAS/LAPACK libraries were not found during installation.\n\n"
                    f"{_format_install_hint()}"
                )

    # Try to import HintSVM
    try:
        from ._hintsvm import hintsvm_query
        from .hintsvm import HintSVM
        _OPTIONAL_FEATURES['hintsvm'] = True
    except ImportError:
        # Create a placeholder class that raises an informative error
        class HintSVM:
            """HintSVM query strategy (not available - requires BLAS/LAPACK)."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "HintSVM is not available because the C extension was not built.\n"
                    "This usually means BLAS/LAPACK libraries were not found during installation.\n\n"
                    f"{_format_install_hint()}"
                )
else:
    # On ReadTheDocs, import the Python wrappers for documentation
    from .variance_reduction import VarianceReduction
    from .hintsvm import HintSVM
    _OPTIONAL_FEATURES['variance_reduction'] = True
    _OPTIONAL_FEATURES['hintsvm'] = True


def get_available_features():
    """
    Return a dictionary of optional features and their availability.

    Returns
    -------
    dict
        Dictionary with feature names as keys and boolean availability as values.
        Example: {'variance_reduction': True, 'hintsvm': False}
    """
    return _OPTIONAL_FEATURES.copy()


def check_optional_features(verbose=True):
    """
    Check and report the availability of optional features.

    Parameters
    ----------
    verbose : bool, default=True
        If True, print the status of each optional feature.

    Returns
    -------
    dict
        Dictionary with feature names as keys and boolean availability as values.
    """
    features = get_available_features()

    if verbose:
        print("libact Optional Features Status")
        print("=" * 40)
        for feature, available in features.items():
            status = "✓ Available" if available else "✗ Not available"
            print(f"  {feature}: {status}")
        print("=" * 40)

        if not all(features.values()):
            print("\nTo enable missing features, install BLAS/LAPACK libraries:")
            print(_format_install_hint())

    return features


__all__ = [
    # Core strategies
    'ActiveLearningByLearning',
    'DWUS',
    'QUIRE',
    'QueryByCommittee',
    'RandomSampling',
    'UncertaintySampling',
    'DensityWeightedMeta',
    # Optional strategies
    'HintSVM',
    'VarianceReduction',
    # Utility functions
    'get_available_features',
    'check_optional_features',
]
