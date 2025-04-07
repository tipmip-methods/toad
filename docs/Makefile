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
	make html
	@echo ">> Documentation built."
	@echo ">> Removing tutorial files..."
	rm -r $(TUTORIAL_DEST)
	@echo ">> ... tutorial files removed."
	@echo ">> Copy content of docs/build/html to a folder called 'docs/build/docs' in the root directory..."
	mkdir -p ./build/docs
	cp -r $(BUILDDIR)/html/* ./build/docs
	@echo ">> ... content copied."
	@echo ">> Delete all other content of ./build..."
	find ./build -type f,d -not -wholename './build/docs*' -not -wholename './build' -not -name '.' -delete
	@echo ">> ... content deleted."
	@echo ">> Script done. Good Bye!"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
