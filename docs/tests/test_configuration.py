"""Regression tests for documentation imports and metadata."""

from pathlib import Path
import sys
import tomllib
import unittest

import yaml


DOCS_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))
sys.path.insert(0, str(REPOSITORY_ROOT))

from native_extensions import (
    NATIVE_EXTENSION_SYMBOLS,
    install_native_extension_stubs,
)
from project_metadata import ensure_version_match, load_project_version
from source_imports import prefer_repository_libact


prefer_repository_libact(REPOSITORY_ROOT)


class DocumentationConfigurationTests(unittest.TestCase):
    def test_native_stubs_are_exact_and_wrappers_remain_real(self):
        installed = install_native_extension_stubs(REPOSITORY_ROOT)

        self.assertEqual(set(NATIVE_EXTENSION_SYMBOLS), set(installed))
        for module_name, symbol_name in NATIVE_EXTENSION_SYMBOLS.items():
            module = sys.modules[module_name]
            self.assertEqual(module.__all__, (symbol_name,))
            self.assertTrue(callable(getattr(module, symbol_name)))
            self.assertFalse(hasattr(module, "arbitrary_missing_symbol"))

        import libact
        from libact.query_strategies import HintSVM, VarianceReduction
        from libact.query_strategies.multiclass.mdsp import MDSP

        self.assertEqual(
            Path(libact.__file__).resolve(),
            (REPOSITORY_ROOT / "libact" / "__init__.py").resolve(),
        )
        self.assertEqual(
            HintSVM.__module__, "libact.query_strategies.hintsvm"
        )
        self.assertEqual(
            VarianceReduction.__module__,
            "libact.query_strategies.variance_reduction",
        )
        self.assertEqual(
            MDSP.__module__, "libact.query_strategies.multiclass.mdsp"
        )

    def test_competing_libact_editable_finder_is_removed(self):
        class CompetingEditableFinder:
            _name = "libact"
            _top_level_modules = {"libact"}

        finder = CompetingEditableFinder()
        sys.meta_path.insert(0, finder)
        prefer_repository_libact(REPOSITORY_ROOT)

        self.assertNotIn(finder, sys.meta_path)

    def test_runtime_requirements_match_package_metadata(self):
        pyproject = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        runtime_requirements = {
            line.strip()
            for line in (DOCS_DIR / "runtime-requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.startswith("#")
        }

        self.assertEqual(
            set(pyproject["project"]["dependencies"]), runtime_requirements
        )

    def test_version_is_loaded_from_metadata_and_mismatch_is_rejected(self):
        pyproject = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        metadata_version = pyproject["project"]["version"]

        self.assertEqual(
            load_project_version(REPOSITORY_ROOT), metadata_version
        )
        with self.assertRaisesRegex(RuntimeError, "Package version mismatch"):
            ensure_version_match(metadata_version, f"{metadata_version}.drift")

    def test_read_the_docs_uses_the_canonical_build(self):
        configuration = yaml.safe_load(
            (REPOSITORY_ROOT / ".readthedocs.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(configuration["version"], 2)
        self.assertEqual(configuration["build"]["os"], "ubuntu-24.04")
        self.assertEqual(
            configuration["build"]["tools"]["python"], "3.12"
        )
        self.assertEqual(
            configuration["python"]["install"],
            [{"requirements": "docs/requirements.txt"}],
        )
        self.assertTrue(configuration["sphinx"]["fail_on_warning"])
        self.assertEqual(
            configuration["build"]["jobs"]["build"]["html"],
            ["make -C docs html BUILDDIR=$READTHEDOCS_OUTPUT"],
        )


if __name__ == "__main__":
    unittest.main()
