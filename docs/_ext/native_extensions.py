"""Narrow import placeholders for libact's optional native extensions."""

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import sys
from types import ModuleType


NATIVE_EXTENSION_SYMBOLS = {
    "libact.query_strategies._hintsvm": "hintsvm_query",
    "libact.query_strategies._variance_reduction": "estVar",
}


def _compiled_module_exists(package_dir, module_name):
    basename = module_name.rsplit(".", 1)[-1]
    return any((package_dir / f"{basename}{suffix}").is_file()
               for suffix in EXTENSION_SUFFIXES)


def _unavailable_callable(module_name, symbol_name):
    def unavailable(*args, **kwargs):
        raise RuntimeError(
            f"{module_name}.{symbol_name} is unavailable because the optional "
            "native extension was not built"
        )

    unavailable.__name__ = symbol_name
    unavailable.__qualname__ = symbol_name
    unavailable.__module__ = module_name
    return unavailable


def install_native_extension_stubs(repository_root):
    """Install exact-symbol stubs only when the native modules are absent."""
    package_dir = Path(repository_root) / "libact" / "query_strategies"
    installed = []

    for module_name, symbol_name in NATIVE_EXTENSION_SYMBOLS.items():
        if module_name in sys.modules:
            continue
        if _compiled_module_exists(package_dir, module_name):
            continue

        module = ModuleType(module_name)
        module.__doc__ = (
            "Documentation-build placeholder for an optional native module."
        )
        module.__all__ = (symbol_name,)
        setattr(
            module,
            symbol_name,
            _unavailable_callable(module_name, symbol_name),
        )
        sys.modules[module_name] = module
        installed.append(module_name)

    return tuple(installed)
