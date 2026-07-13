"""Load and validate package metadata used by Sphinx."""

from pathlib import Path
import tomllib


def ensure_version_match(metadata_version, package_version):
    """Reject drift between packaging metadata and the importable package."""
    if metadata_version != package_version:
        raise RuntimeError(
            "Package version mismatch: pyproject.toml declares "
            f"{metadata_version!r}, but libact.__version__ is "
            f"{package_version!r}"
        )


def load_project_version(repository_root):
    """Return the authoritative project version after checking libact."""
    repository_root = Path(repository_root)
    pyproject = tomllib.loads(
        (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    metadata_version = pyproject["project"]["version"]

    import libact

    package_file = Path(libact.__file__).resolve()
    expected_package_dir = (repository_root / "libact").resolve()
    if package_file.parent != expected_package_dir:
        raise RuntimeError(
            "Documentation imported libact from "
            f"{str(package_file)!r}; expected source under "
            f"{str(expected_package_dir)!r}"
        )

    ensure_version_match(metadata_version, libact.__version__)
    return metadata_version
