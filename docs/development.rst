Development
===========

Building the documentation
--------------------------

The documentation environment uses Python 3.12 and does not install libact or
compile its optional native extensions. From a clean checkout, create an
isolated environment and install the declared dependencies:

.. code-block:: console

   python3.12 -m venv .venv-docs
   . .venv-docs/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r docs/requirements.txt

Build strict HTML documentation with the canonical command:

.. code-block:: console

   make -C docs html

The generated site is written to ``docs/_build/html``. Warnings, broken
internal references, and autodoc import failures make this command fail.

Clean generated files before rebuilding:

.. code-block:: console

   make -C docs clean
   make -C docs html

Run the same regression tests, clean build, and output checks used by CI:

.. code-block:: console

   make -C docs check

External links can be checked separately because remote sites may be
temporarily unavailable:

.. code-block:: console

   make -C docs linkcheck

Development documentation
-------------------------

.. toctree::
   :maxdepth: 1

   dev_with_libact
   documentation_follow_up
