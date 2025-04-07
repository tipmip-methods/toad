# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = ./source
BUILDDIR      = ./build
TUTORIAL_SRC  = ./../tutorials
TUTORIAL_DEST = ./source/tutorials

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help toad Makefile

# custom html maker:
# 	- copies the tutorial files to the source directory
# 	- builds the documentation
# 	- removes the tutorial files from source directory
toad:
	@echo ">> Copying tutorial files from $(TUTORIAL_SRC) to $(TUTORIAL_DEST)"
	cp -r $(TUTORIAL_SRC) $(TUTORIAL_DEST)
	@echo ">> Files copied."
	@echo ">> Deleting build directory..."
	rm -r $(BUILDDIR)
	@echo ">> Build directory deleted."
	@echo ">> Building documentation..."
	make html > build.log 2>&1
	@echo ">> Documentation built."
	@echo ">> Removing tutorial files..."
	rm -r $(TUTORIAL_DEST)
	@echo ">> Tutorial files removed."
	@echo ">> Script done. Good Bye!"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
