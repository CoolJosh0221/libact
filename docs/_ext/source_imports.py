"""Keep autodoc bound to the libact source in this checkout."""

from pathlib import Path
import sys


def _targets_libact(finder):
    if getattr(finder, "_name", None) == "libact":
        return True

    top_level_modules = getattr(finder, "_top_level_modules", ())
    if "libact" in top_level_modules:
        return True

    mapping = getattr(finder, "MAPPING", {})
    return isinstance(mapping, dict) and "libact" in mapping


def _is_from_package(module, package_dir):
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False

    try:
        Path(module_file).resolve().relative_to(package_dir)
    except ValueError:
        return False
    return True


def prefer_repository_libact(repository_root):
    """Prefer this checkout over competing editable libact installations."""
    repository_root = Path(repository_root).resolve()
    package_dir = repository_root / "libact"

    sys.meta_path[:] = [
        finder for finder in sys.meta_path if not _targets_libact(finder)
    ]

    for module_name, module in tuple(sys.modules.items()):
        if module_name == "libact" or module_name.startswith("libact."):
            if not _is_from_package(module, package_dir):
                del sys.modules[module_name]

    repository_path = str(repository_root)
    sys.path[:] = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != repository_root
    ]
    sys.path.insert(0, repository_path)
