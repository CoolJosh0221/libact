SPHINXOPTS ?= -W --keep-going -n
SPHINXBUILD ?= sphinx-build
PYTHON ?= python
SOURCEDIR = .
BUILDDIR ?= _build
HTMLDIR = $(BUILDDIR)/html

export MPLCONFIGDIR ?= $(abspath $(BUILDDIR)/matplotlib)

.PHONY: help clean html linkcheck test verify check

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

linkcheck:
	$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

test:
	$(PYTHON) -m unittest discover -s tests -v

verify:
	test -s "$(HTMLDIR)/index.html"
	test -s "$(HTMLDIR)/api_reference.html"
	test -s "$(HTMLDIR)/libact.query_strategies.html"
	grep -q "HintSVM" "$(HTMLDIR)/libact.query_strategies.html"
	grep -q "VarianceReduction" "$(HTMLDIR)/libact.query_strategies.html"

check:
	+$(MAKE) clean
	+$(MAKE) test
	+$(MAKE) html
	+$(MAKE) verify
