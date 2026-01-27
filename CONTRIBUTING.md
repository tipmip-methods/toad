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
3. **`CHANGELOG.md`**: Fill in changes. 

Then: 
1. Commit these changes with the commit message like `Release v1.0.1`:
   ```bash
   git add toad/_version.py CITATION.cff CHANGELOG.md
   git commit -m "Release v1.0.1"
   ```
2. Push the commit to `main`:
   ```bash
   git push origin main
   ```
3. Create a git tag like `v1.0.1` and push it:
   ```bash
   git tag v1.0.1
   git push origin v1.0.1
   ```
   A Github Action Workflow [(create-draft-release.yml)](https://github.com/tipmip-methods/toad/blob/main/.github/workflows/create-draft-release.yml) will automatically create a **draft** GitHub release with a link to `CHANGELOG.md`.

4. Review and publish the draft release on GitHub.com:
   - Go to the [Releases page](https://github.com/tipmip-methods/toad/releases)
   - Find the draft release
   - Optionally edit the release notes if needed
   - Click "Publish release"
   
   Publishing the release will:
   - Trigger the PyPI publishing workflow [(publish-to-pypi.yml)](https://github.com/tipmip-methods/toad/blob/main/.github/workflows/publish-to-pypi.yml) automatically
   - Trigger Zenodo to archive the release and generate a new DOI
