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

    from libact import __version__ as package_version

    ensure_version_match(metadata_version, package_version)
    return metadata_version
