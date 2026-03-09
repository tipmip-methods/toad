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
            assert "cluster_labels" in ds
            assert ds["cluster_labels"].attrs.get("format") == "native"
            assert "lat" in ds.coords or "latitude" in ds.coords
            ds.close()


def test_export_for_mma_healpix():
    """Test compute_clusters export_for_mma with mma_grid='healpix'."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=2, nside=8)
        assert len(paths) == 2
        for p in paths:
            ds = __import__("xarray").open_dataset(p)
            assert "cluster_labels" in ds
            assert ds["cluster_labels"].attrs.get("format") == "healpix"
            assert ds.attrs.get("nside") == 8
            npix = 12 * 8**2
            assert ds["cluster_labels"].shape == (npix,)
            ds.close()


def test_mma_native_files():
    """Test MMA with native-format cluster label files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_native(Path(tmp), n_runs=3)
        mma = MMA(paths, nside=16)
        ds = mma.run_consensus(min_consensus=0.5, min_cluster_size=2)
        assert "consensus_clusters" in ds
        assert "consensus_consistency" in ds
        assert ds["consensus_clusters"].shape == (12 * 16**2,)
        labels = ds["consensus_clusters"].values
        assert np.any(np.isfinite(labels) & (labels >= 0)) or np.any(labels == -1)


def test_mma_healpix_files():
    """Test MMA with HealPix-format cluster label files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = setup_and_export_healpix(Path(tmp), n_runs=3, nside=8)
        mma = MMA(paths, nside=8)
        ds = mma.run_consensus(min_consensus=0.4, min_cluster_size=1)
        assert "consensus_clusters" in ds
        assert "consensus_consistency" in ds
        assert ds["consensus_clusters"].shape == (12 * 8**2,)
