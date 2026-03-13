"""Tests for MMA (multi-model aggregation) pipeline."""

import tempfile
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from toad import MMA, TOAD
from toad.regridding import HealPixRegridder
from toad.shifts import ASDETECT


def setup_and_export_native(tmp_path: Path, n_runs: int = 3) -> list[str]:
    """Create clusterings, export as native format via export_for_mma, return paths."""
    td = TOAD("tutorials/test_data/synth_data.nc", time_dim="time")
    td.data = td.data.coarsen(lat=3, lon=3, time=3, boundary="trim").reduce(np.mean)

    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        td.compute_shifts(td.base_vars[0], method=ASDETECT(ignore_nan_warnings=True))

    paths = []
    for i in range(n_runs):
        p = tmp_path / f"native_{i}.nc"
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=5),
            time_weight=0.5 + i * 0.3,
            shift_threshold=0.8,
            export_for_mma=str(p),
            mma_grid="native",
        )
        paths.append(str(p))
    return paths


def setup_and_export_healpix(
    tmp_path: Path, n_runs: int = 3, nside: int = 16
) -> list[str]:
    """Create clusterings, export as HealPix format via export_for_mma, return paths."""
    td = TOAD("tutorials/test_data/synth_data.nc", time_dim="time")
    td.data = td.data.coarsen(lat=4, lon=4, time=3, boundary="trim").reduce(np.mean)

    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        td.compute_shifts(td.base_vars[0], method=ASDETECT(ignore_nan_warnings=True))

    paths = []
    for i in range(n_runs):
        p = tmp_path / f"healpix_{i}.nc"
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=5),
            time_weight=0.5 + i * 0.25,
            shift_threshold=0.8,
            regridder=HealPixRegridder(nside=nside),
            export_for_mma=str(p),
            mma_grid="healpix",
        )
        paths.append(str(p))
    return paths


def test_export_for_mma_native():
    """Test compute_clusters export_for_mma with mma_grid='native'."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_native(Path(tmp), n_runs=2)
        assert len(paths) == 2
        for p in paths:
            ds = __import__("xarray").open_dataset(p)
            assert "cluster" in ds
            assert "lat" in ds.coords or "latitude" in ds.coords
            ds.close()


def test_export_for_mma_healpix():
    """Test compute_clusters export_for_mma with mma_grid='healpix'."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=2, nside=8)
        assert len(paths) == 2
        for p in paths:
            ds = __import__("xarray").open_dataset(p)
            assert "cluster" in ds
            assert "hp_pixel" in ds["cluster"].dims
            assert ds.attrs.get("nside") == 8
            npix = 12 * 8**2
            assert ds["cluster"].shape == (ds.sizes["time"], npix)
            ds.close()


def test_mma_native_files():
    """Test MMA with native-format cluster label files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_native(Path(tmp), n_runs=3)
        mma = MMA(paths, nside=None)
        rate = mma.cluster_occurrence_rate()
        assert 0 <= float(rate.min()) <= float(rate.max()) <= 1
        ds = mma.run_consensus(min_consensus=0.5, min_cluster_size=2)
        assert "consensus_clusters" in ds
        assert "consensus_consistency" in ds
        assert (
            "lat" in ds["consensus_clusters"].dims
            or "lon" in ds["consensus_clusters"].dims
        )
        labels = ds["consensus_clusters"].values
        assert np.any(np.isfinite(labels) & (labels >= 0)) or np.any(labels == -1)
        # Plot native format (synth_data has lat/lon)
        fig, ax = mma.plot_consensus_clusters(
            map_style={"projection": "plate_carree"},
        )
        assert fig is not None
        assert ax is not None


def test_mma_native_files_rejected_with_nside():
    """Test MMA raises when nside is passed with native-format exports."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_native(Path(tmp), n_runs=2)
        with __import__("pytest").raises(ValueError) as exc_info:
            MMA(paths, nside=16)
        assert "native" in str(exc_info.value).lower()
        assert "nside=None" in str(exc_info.value)


def test_mma_healpix_files():
    """Test MMA with HealPix-format cluster label files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=3, nside=8)
        mma = MMA(paths, nside=8)
        ds = mma.run_consensus(min_consensus=0.4, min_cluster_size=1)
        assert "consensus_clusters" in ds
        assert "consensus_consistency" in ds
        assert ds["consensus_clusters"].shape == (12 * 8**2,)
        # Shift times from HealPix cluster (no native grid in export)
        times_by_cluster = mma.get_shift_times_per_consensus_cluster()
        assert isinstance(times_by_cluster, dict)
        rate = mma.cluster_occurrence_rate()
        assert rate.shape == ds["consensus_clusters"].shape
        assert 0 <= rate.min() <= rate.max() <= 1
        summary = mma.get_consensus_summary()
        assert "cluster_id" in summary.columns
        assert "size" in summary.columns
        assert "mean_consistency" in summary.columns
        assert "mean_mean_shift_time" in summary.columns
        assert "std_mean_shift_time" in summary.columns
