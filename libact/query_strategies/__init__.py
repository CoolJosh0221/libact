"""
Concrete query strategy classes.
"""
from __future__ import absolute_import

from .active_learning_by_learning import ActiveLearningByLearning
from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
from .density_weighted_uncertainty_sampling import DWUS
from .bald import BALD
from .coreset import CoreSet
from .epsilon_uncertainty_sampling import EpsilonUncertaintySampling
from .information_density import InformationDensity
from .density_weighted_meta import DensityWeightedMeta
# HintSVM and VarianceReduction are backed by compiled C extensions that link
# BLAS/LAPACK and are only built on POSIX platforms. The wrapper modules always
# import cleanly; instantiating the strategy raises
# libact.base.exceptions.ExtensionUnavailable when the extension is missing
# (e.g. on a pure-Python Windows install).
from .variance_reduction import VarianceReduction
from .hintsvm import HintSVM

__all__ = [
    'ActiveLearningByLearning',
    'BALD',
    'CoreSet',
    'DWUS',
    'EpsilonUncertaintySampling',
    'HintSVM',
    'InformationDensity',
    'QUIRE',
    'QueryByCommittee',
    'RandomSampling',
    'UncertaintySampling',
    'VarianceReduction',
    'DensityWeightedMeta',
]
