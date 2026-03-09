"""Multi-model aggregation (MMA) for consensus clustering across saved cluster label files."""

import logging
from typing import List

import numpy as np
import xarray as xr

from toad._version import __version__
from toad.regridding.healpix import HealPixRegridder
from toad.utils import _attrs, detect_latlon_names
from toad.utils.cluster_consensus_utils import run_healpix_consensus

logger = logging.getLogger("TOAD")


def _load_cluster_labels(path: str) -> tuple[np.ndarray, int | None]:
    """Load cluster labels from a netCDF file.

    Returns:
        Tuple of (labels_array, nside).
        - Native format: labels shape (y, x), nside=None.
        - HealPix format: labels shape (npix,), nside=int.
    """
    ds = xr.open_dataset(path)
    labels = ds["cluster_labels"].values
    attrs = ds["cluster_labels"].attrs
    fmt = attrs.get("format", "native")
    ds.close()

    if fmt == "healpix":
        nside = int(attrs.get("nside"))
        return labels.astype(np.float32), nside
    else:
        return labels.astype(np.float32), None


def _native_to_healpix(
    labels_2d: np.ndarray, lat: np.ndarray, lon: np.ndarray, nside: int
) -> np.ndarray:
    """Regrid native-grid cluster labels to HealPix."""
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    regridder = HealPixRegridder(nside=nside)
    coords_2d = np.column_stack([lat.ravel(), lon.ravel()])
    hp_index = regridder.map_orig_to_regrid(coords_2d)

    npix = 12 * nside**2
    hp_label_counts: dict[int, dict[float, int]] = {}
    flat_labels = labels_2d.ravel()
    for flat_idx in range(flat_labels.size):
        lbl = flat_labels[flat_idx]
        if not (np.isfinite(lbl) or lbl == -1):
            continue
        hp_idx = int(hp_index[flat_idx])
        if hp_idx not in hp_label_counts:
            hp_label_counts[hp_idx] = {}
        k = float(lbl)
        hp_label_counts[hp_idx][k] = hp_label_counts[hp_idx].get(k, 0) + 1
    hp_labels = np.full(npix, np.nan, dtype=np.float32)
    for hp_idx, counts in hp_label_counts.items():
        hp_labels[hp_idx] = max(counts.items(), key=lambda x: x[1])[0]
    return hp_labels


class MMA:
    """Multi-model aggregation: consensus clustering across cluster label files.

    Loads cluster label files (saved via ``TOAD.save_cluster_labels``), regrids
    to a common HealPix grid if needed, and runs consensus clustering.
    """

    def __init__(
        self,
        paths: List[str],
        nside: int = 32,
    ):
        """Load cluster label files for consensus.

        Args:
            paths: List of paths to netCDF files with cluster labels
                (from TOAD.save_cluster_labels).
            nside: HealPix nside for the common grid. Native-format files are
                regridded to this; HealPix files must already use this nside.

        Raises:
            ValueError: If no paths, or if HealPix files have wrong nside.
        """
        if not paths:
            raise ValueError("paths must not be empty.")
        self.paths = [str(p) for p in paths]
        self.nside = nside
        self._hp_arrays: list[np.ndarray] = []
        self._data: xr.Dataset | None = None
        self._load_all()

    def _load_all(self) -> None:
        for path in self.paths:
            labels, nside_in = _load_cluster_labels(path)
            if nside_in is not None:
                if nside_in != self.nside:
                    raise ValueError(
                        f"File {path} has nside={nside_in}, but MMA uses nside={self.nside}. "
                        "All files must use the same nside."
                    )
                self._hp_arrays.append(labels)
            else:
                # Native format: need lat/lon to regrid
                ds = xr.open_dataset(path)
                lat_name, lon_name = detect_latlon_names(ds)
                if lat_name is None or lon_name is None:
                    ds.close()
                    raise ValueError(
                        f"Native-format file {path} must have lat/lon coordinates."
                    )
                lat = ds[lat_name].values
                lon = ds[lon_name].values
                ds.close()
                hp_labels = _native_to_healpix(labels, lat, lon, self.nside)
                self._hp_arrays.append(hp_labels)
        logger.info(
            f"MMA: loaded {len(self._hp_arrays)} cluster label file(s), nside={self.nside}"
        )

    def run_consensus(
        self,
        min_consensus: float = 0.5,
        min_cluster_size: int = 1,
        k_neighbors: int = 8,
        top_n_clusters: int | None = None,
        show_progress: bool = False,
    ) -> xr.Dataset:
        """Run consensus clustering and store results in self.data.

        Args:
            min_consensus: Minimum consensus threshold in [0, 1].
            min_cluster_size: Minimum cluster size; smaller demoted to -1.
            k_neighbors: K for KNN graph on HealPix.
            top_n_clusters: If set, only consider top N clusters per model.
            show_progress: Whether to show progress.

        Returns:
            The internal dataset with consensus_clusters and consensus_consistency.
        """
        labels, consistency = run_healpix_consensus(
            self._hp_arrays,
            self.nside,
            min_consensus=min_consensus,
            min_cluster_size=min_cluster_size,
            k_neighbors=k_neighbors,
            top_n_clusters=top_n_clusters,
            show_progress=show_progress,
        )
        npix = 12 * self.nside**2

        shared_attrs = {
            "min_consensus": min_consensus,
            "min_cluster_size": min_cluster_size,
            "top_n_clusters": top_n_clusters,
            "k_neighbors": k_neighbors,
            "nside": self.nside,
            "n_models": len(self._hp_arrays),
            "source_paths": self.paths,
            _attrs.METHOD_NAME: "mma_consensus",
            _attrs.TOAD_VERSION: __version__,
        }

        da_clusters = xr.DataArray(
            labels,
            dims=["hp_pixel"],
            coords={"hp_pixel": np.arange(npix)},
            name="consensus_clusters",
            attrs={
                **shared_attrs,
                "description": "Spatial consensus clusters on HealPix (time-collapsed).",
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CLUSTER,
                _attrs.CONSENSUS_CONSISTENCY_VARIABLE: "consensus_consistency",
            },
        )
        da_consistency = xr.DataArray(
            consistency,
            dims=["hp_pixel"],
            coords={"hp_pixel": np.arange(npix)},
            name="consensus_consistency",
            attrs={
                **shared_attrs,
                "description": "Consistency scores for each HealPix pixel.",
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CONSISTENCY,
            },
        )

        self.data = xr.merge(
            [da_clusters, da_consistency],
            combine_attrs="override",
            compat="override",
        )
        self.data.attrs["nside"] = self.nside
        self.data.attrs["format"] = "healpix"

        n_clusters = len(np.unique(labels[(labels >= 0) & np.isfinite(labels)]))
        logger.info(
            f"MMA consensus: {n_clusters} clusters, "
            f"noise={int(np.sum(labels == -1))}, "
            f"nan={int(np.sum(np.isnan(labels)))}"
        )
        return self.data

    @property
    def data(self) -> xr.Dataset:
        """Internal dataset with consensus_clusters and consensus_consistency."""
        if self._data is None:
            raise AttributeError(
                "Run run_consensus() first to compute and store results."
            )
        return self._data

    @data.setter
    def data(self, value: xr.Dataset) -> None:
        self._data = value
