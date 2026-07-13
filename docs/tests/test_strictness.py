"""Exercise strict Sphinx behavior against controlled documentation defects."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


DOCS_DIR = Path(__file__).resolve().parents[1]


class StrictSphinxBuildTests(unittest.TestCase):
    def _build(self, body):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            (source / "index.rst").write_text(
                textwrap.dedent(body), encoding="utf-8"
            )

            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = str(root / "matplotlib")
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sphinx",
                    "-q",
                    "-W",
                    "--keep-going",
                    "-n",
                    "-c",
                    str(DOCS_DIR),
                    str(source),
                    str(output),
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )
            return process

    def test_native_wrappers_and_ordinary_python_class_build(self):
        process = self._build(
            """
            Import boundary
            ===============

            .. autoclass:: libact.query_strategies.HintSVM

            .. autoclass:: libact.query_strategies.VarianceReduction

            .. autoclass:: libact.query_strategies.UncertaintySampling

            .. automodule:: libact.base.interfaces
               :members:
            """
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_missing_ordinary_libact_module_fails(self):
        process = self._build(
            """
            Broken module
            =============

            .. automodule:: libact.documentation_target_that_does_not_exist
            """
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(
            "libact.documentation_target_that_does_not_exist", process.stderr
        )

    def test_missing_public_object_in_autosummary_fails(self):
        process = self._build(
            """
            Broken object
            =============

            .. currentmodule:: libact.query_strategies

            .. autosummary::

               DocumentationTargetThatDoesNotExist
            """
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("DocumentationTargetThatDoesNotExist", process.stderr)

    def test_broken_internal_document_reference_fails(self):
        process = self._build(
            """
            Broken document reference
            =========================

            See :doc:`documentation-page-that-does-not-exist`.
            """
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(
            "documentation-page-that-does-not-exist", process.stderr
        )

    def test_duplicate_document_label_fails(self):
        process = self._build(
            """
            .. _duplicate-document-label:

            First label
            ===========

            .. _duplicate-document-label:

            Second label
            ============
            """
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("duplicate-document-label", process.stderr)

    def test_missing_static_asset_fails(self):
        process = self._build(
            """
            Missing image
            =============

            .. image:: image-that-does-not-exist.png
            """
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("image-that-does-not-exist.png", process.stderr)


if __name__ == "__main__":
    unittest.main()
