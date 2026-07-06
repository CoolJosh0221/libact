"""Tests for graceful degradation when compiled C extensions are unavailable.

HintSVM and VarianceReduction are backed by compiled extensions that link
BLAS/LAPACK and are only built on POSIX platforms. When the extension is not
available, importing the strategy class still works, but instantiating it must
raise ExtensionUnavailable (a subclass of ImportError) with a helpful message --
never a bare "DLL load failed" ImportError.

These tests simulate a missing extension by monkeypatching the module-level
symbol to None, so they run identically whether or not the extension is compiled.
"""
import unittest
from unittest import mock

from libact.base.dataset import Dataset
from libact.base.exceptions import ExtensionUnavailable
# The wrapper classes must always be importable, even without the extensions.
from libact.query_strategies import HintSVM, VarianceReduction
from libact.query_strategies import hintsvm as hintsvm_module
from libact.query_strategies import variance_reduction as variance_reduction_module


class ExtensionUnavailableTestCase(unittest.TestCase):

    def setUp(self):
        X = [[-2, -1], [1, 1], [-1, -2], [2, 1]]
        y = [-1, 1, None, None]
        self.dataset = Dataset(X, y)

    def test_extension_unavailable_is_importerror(self):
        self.assertTrue(issubclass(ExtensionUnavailable, ImportError))

    @mock.patch.object(hintsvm_module, 'hintsvm_query', None)
    def test_hintsvm_raises_when_extension_missing(self):
        with self.assertRaises(ExtensionUnavailable) as ctx:
            HintSVM(self.dataset)
        message = str(ctx.exception)
        self.assertIn('HintSVM', message)
        self.assertIn('_hintsvm', message)
        self.assertIn('Windows', message)
        # ExtensionUnavailable must be catchable as ImportError.
        with self.assertRaises(ImportError):
            HintSVM(self.dataset)

    @mock.patch.object(variance_reduction_module, 'estVar', None)
    def test_variance_reduction_raises_when_extension_missing(self):
        with self.assertRaises(ExtensionUnavailable) as ctx:
            VarianceReduction(self.dataset)
        message = str(ctx.exception)
        self.assertIn('VarianceReduction', message)
        self.assertIn('_variance_reduction', message)
        self.assertIn('Windows', message)
        with self.assertRaises(ImportError):
            VarianceReduction(self.dataset)


if __name__ == '__main__':
    unittest.main()
