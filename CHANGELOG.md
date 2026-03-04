# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Stats can now be accessed as a property (e.g. `td.stats.time.start(cluster_id=0)`) when using a single cluster variable
- Function for removing specific clusters by id (useful for filtering clusters of no interest)
- Function for sorting clusters (e.g. by magnitude, median shift time, or to reset indexing after removal)
- New time stats: `value_at_start`, `value_at_end`, `value_change`, `value_at_iqr_90_start`/`value_at_iqr_90_end`, `value_change_iqr_90`

### Fixed
- More robust variable selection in consensus clustering (cluster variable is now specified directly instead of inferred from shifts)
- `compute_transition_time` now restricted to cluster period (previously used largest shift in each grid cell regardless of period)
- Z-order of contours in cluster maps (clusters now correctly layered)
- Enhanced error handling in asdetect for short time series (validates `lmin` < `lmax` with guiding error message for common defaults)
- Assert that all variables have the same dimensions (fixes inference from first base variable in `space_dims`)

## [1.0.5] - 2026-02-12

### Added
- Allow custom cluster methods to turn off temporal scaling by adding `skip_time_scaling = True` as a class variable to the method.
- Allow disabling the gradient legend in cluster maps (set `other_legend=False` in `MapStyle`)
- Added progress bar for cluster consensus computation (`show_progress` parameter)
- Added comprehensive tests for plotting and postprocessing functions

### Changed
- Cluster output now distinguishes between grid cells classified as noise by the clustering algorithm (-1) and grid cells with no detected abrupt shifts (NaN). Previously both were labeled -1.
- Enhanced variable inference in TOAD methods
- Relaxed dependency version pins for numpy, scipy, and scikit-learn (removed upper bounds)

### Fixed
- Fixed base variable inference in `get_cluster_timeseries` for scoring functions
- Fixed dependency configuration: added `tqdm` as direct dependency, corrected deptry module name mappings and ignore lists

## [1.0.4] - 2026-01-27

### Changed
- Updated README.md installation instructions to include pip installation command. 

## [1.0.3] - 2026-01-27

### Fixed
- Fixed image paths in README.md to use absolute URLs for proper display on PyPI

## [1.0.2] - 2026-01-27

### Changed
- Updated PyPI publishing workflow to use Trusted Publishers (OIDC) instead of API tokens for improved security

## [1.0.1] - 2026-01-27

### Changed
- Updated README.md with improved installation and usage instructions
- Enhanced documentation structure and content
- Reorganized and updated existing tutorials

### Added
- Added CITATION.cff for citation support
- Added CONTRIBUTING.md with contribution guidelines
- Added GitHub Actions workflow for publishing to PyPI — first PyPI release!
- Added Github Actions workflow for automatically creating draft release when tag with v* is pushed to main. 

## [1.0.0] - 2026-01-22

### Added
- First public release of the TOAD package

[Unreleased]: https://github.com/tipmip-methods/toad/compare/v1.0.5...HEAD
[1.0.5]: https://github.com/tipmip-methods/toad/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/tipmip-methods/toad/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/tipmip-methods/toad/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/tipmip-methods/toad/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/tipmip-methods/toad/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tipmip-methods/toad/releases/tag/v1.0.0
