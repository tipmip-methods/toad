# Contributing to TOAD

Thank you for your interest in contributing to TOAD! This document outlines our development workflow and coding standards.

## Development Setup

After cloning the repository, install the package in editable mode with development dependencies:

```bash
$ git clone https://github.com/tipmip-methods/toad.git
$ cd toad
$ pip install -e .[dev]
```

The `-e` flag installs the package in "editable" mode, which means changes to the source code are immediately reflected without needing to reinstall.

## Code Formatting

We use [ruff](https://github.com/astral-sh/ruff) for consistent code formatting across the project.

### Pre-commit hooks

To ensure consistent code formatting and quality, we use pre-commit hooks. After installing the dev dependencies, install the pre-commit hooks:

```bash
$ pre-commit install
```

The hooks will automatically run on each commit, checking and fixing code formatting with ruff. If you want to manually run the hooks on all files:

```bash
$ pre-commit run --all-files
```

## Development Workflow

We use [trunk-based development](https://medium.com/@vafrcor2009/gitflow-vs-trunk-based-development-3beff578030b) for our git workflow. This means we all work on the same branch (main), the trunk, and push our code changes to it often. This way, we can keep our code up to date. We also avoid having too many branches that can get messy and hard to merge. We only create short-lived branches for small features or bug fixes, and we merge them back to main as soon as they are done. To this end, each developer issues pull-requests that are approved or rejected by the maintainer. Special versions of the code can be then dedicated releases with version tags, allowing others to use very specific versions of the code if needed.

## Releasing a New Version

When creating a new release, maintainers should update the following files:

1. **`toad/_version.py`**: Update `__version__` to the new version number (e.g., `"1.0.1"`)
2. **`CITATION.cff`**: Update:
   - `version`: Set to the new version number
   - `date-released`: Set to the release date (format: `YYYY-MM-DD`)
Note: `doi` always points to the latest release. 

After updating these files, create a git tag with the version number (e.g. `v1.0.1`) and push it to trigger the release process.
