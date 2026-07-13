"""Sphinx configuration for the libact documentation."""

from pathlib import Path
import sys


DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
EXTENSIONS_DIR = DOCS_DIR / "_ext"

sys.path.insert(0, str(EXTENSIONS_DIR))
sys.path.insert(0, str(REPOSITORY_ROOT))

from native_extensions import install_native_extension_stubs
from project_metadata import load_project_version


# Keep autodoc on the real source tree while providing only the two symbols
# whose compiled implementations are intentionally absent from docs builds.
install_native_extension_stubs(REPOSITORY_ROOT)

project = "libact"
author = (
    "Y.-Y. Yang, S.-C. Lee, Y.-A. Chung, T.-E. Wu, S.-A. Chen, "
    "H.-T. Lin"
)
copyright = f"2015-%Y, {author}"
release = load_project_version(REPOSITORY_ROOT)
version = release

needs_sphinx = "9.1"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]

source_suffix = {".rst": "restructuredtext"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = []

nitpicky = True
autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autodoc_typehints = "description"

numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_xref_param_type = False

modindex_common_prefix = ["libact."]
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_title = f"libact {release} documentation"
html_static_path = []
htmlhelp_basename = "libactdoc"
