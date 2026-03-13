"""Multi-model aggregation (MMA) for consensus clustering across saved cluster label files."""

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toad._version import __version__

if TYPE_CHECKING:
    from toad.plotting import MapStyle
from toad.regridding.healpix import HealPixRegridder
from toad.utils import _attrs, detect_latlon_names
from toad.utils.cluster_consensus_utils import (
    _build_knn_edges_from_coords_2d,
    run_healpix_consensus,
    run_native_consensus,
)

logger = logging.getLogger("TOAD")


def _get_native_spatial_coords(
    ds: xr.Dataset, cluster: xr.DataArray
) -> tuple[np.ndarray, bool]:
    """Get (N, 2) spatial coords for KNN. Tries lat/lon first, then x/y.

    Returns:
        coords_2d: (n_spatial, 2) array for KNN.
        use_sphere: True if lat/lon (use spherical distance).
    """
    spatial_dims = list(cluster.dims[1:])  # skip time dim
    if len(spatial_dims) < 2:
        raise ValueError("Native cluster must have at least 2 spatial dimensions.")

    lat_name, lon_name = detect_latlon_names(ds)
    if (
        lat_name is not None
        and lon_name is not None
        and lat_name in ds
        and lon_name in ds
    ):
        c1 = np.asarray(ds[lat_name].values)
        c2 = np.asarray(ds[lon_name].values)
        use_sphere = True
    else:
        # Get coords for each spatial dim (x,y or dim index)
        coords_for_dims: list[np.ndarray] = []
        for d in spatial_dims:
            if d in ds.coords:
                coords_for_dims.append(np.asarray(ds[d].values))
            elif d in ds:
                coords_for_dims.append(np.asarray(ds[d].values))
            else:
                # Fallback: use arange for this dim
                coords_for_dims.append(np.arange(cluster.sizes[d], dtype=np.float64))
        c1, c2 = coords_for_dims[0], coords_for_dims[1]
        use_sphere = False

    if c1.ndim == 1 and c2.ndim == 1:
        c2g, c1g = np.meshgrid(c2, c1)
    else:
        c1g = c1
        c2g = c2
    coords_2d = np.column_stack([c1g.ravel(), c2g.ravel()])
    return coords_2d, use_sphere


def _load_ever_in_healpix_masks(path: str, nside: int) -> list[tuple[int, np.ndarray]]:
    """Load cluster data from HealPix export and compute ever-in masks."""
    ds = xr.open_dataset(path)
    if "cluster" not in ds:
        ds.close()
        raise ValueError(
            f"Export {path} has no 'cluster' variable. "
            "Re-export with compute_clusters(export_for_mma=...)."
        )
    cluster = ds["cluster"]
    npix = 12 * nside**2
    if "hp_pixel" not in cluster.dims:
        ds.close()
        raise ValueError(
            f"Export {path} has native-format cluster (no hp_pixel dim). "
            "Use nside=None for MMA with native exports."
        )
    arr = np.asarray(cluster.values, dtype=np.float64)
    ds.close()
    if arr.shape[1] != npix:
        raise ValueError(
            f"cluster has {arr.shape[1]} pixels, expected {npix} for nside={nside}."
        )
    result: list[tuple[int, np.ndarray]] = []
    valid = (arr >= 0) & np.isfinite(arr)
    unique_cids = np.unique(arr[valid].astype(np.int64))
    for cid in unique_cids:
        mask = ((arr == cid) & valid).any(axis=0)
        if mask.any():
            result.append((int(cid), mask))
    return result


def _load_ever_in_native_masks(
    path: str,
) -> tuple[
    list[tuple[int, np.ndarray]],
    np.ndarray,
    bool,
    list[str],
    tuple[int, ...],
]:
    """Load cluster data from native export and compute ever-in masks.

    Returns:
        masks, coords_2d, use_sphere, spatial_dims, spatial_shape
    """
    ds = xr.open_dataset(path)
    if "cluster" not in ds:
        ds.close()
        raise ValueError(
            f"Export {path} has no 'cluster' variable. "
            "Re-export with compute_clusters(export_for_mma=...)."
        )
    cluster = ds["cluster"]
    if "hp_pixel" in cluster.dims:
        ds.close()
        raise ValueError(
            f"Export {path} has HealPix cluster. Use nside=... for MMA with HealPix exports."
        )
    arr = np.asarray(cluster.values, dtype=np.float64)
    # First dim is time, rest are spatial
    spatial_dims = [str(d) for d in cluster.dims[1:]]
    spatial_shape = tuple(cluster.sizes[d] for d in cluster.dims[1:])
    coords_2d, use_sphere = _get_native_spatial_coords(ds, cluster)
    ds.close()

    result: list[tuple[int, np.ndarray]] = []
    cluster_flat = arr.reshape(arr.shape[0], -1)
    valid = (cluster_flat >= 0) & np.isfinite(cluster_flat)
    unique_cids = np.unique(cluster_flat[valid].astype(np.int64))
    for cid in unique_cids:
        mask = ((cluster_flat == cid) & valid).any(axis=0)
        if mask.any():
            result.append((int(cid), mask))
    return result, coords_2d, use_sphere, spatial_dims, spatial_shape


class MMA:
    """Multi-model aggregation: consensus clustering across cluster label files.

    Loads cluster label files (HealPix or native format) exported via
    ``compute_clusters(export_for_mma=...)`` and runs consensus clustering.
    Supports both hp_pixel (HealPix) and native (x,y or lat/lon) grids.
    """

    def __init__(
        self,
        paths: List[str],
        nside: Optional[int] = 32,
    ):
        """Load cluster label files for consensus.

        Args:
            paths: List of paths to netCDF files with cluster labels
                (from compute_clusters(export_for_mma=...)).
            nside: HealPix nside for HealPix exports; use None for native exports
                (x,y or lat/lon grids). Format is auto-detected from the first file.

        Raises:
            ValueError: If no paths, or if format/nside mismatch.
        """
        if not paths:
            raise ValueError("paths must not be empty.")
        self.paths = [str(p) for p in paths]
        self.nside = nside
        self._cluster_masks: list[list[tuple[int, np.ndarray]]] = []
        self._format: str = "healpix"  # or "native"
        self._native_coords_2d: Optional[np.ndarray] = None
        self._native_use_sphere: bool = False
        self._native_spatial_dims: Optional[List[str]] = None
        self._native_spatial_shape: Optional[tuple[int, ...]] = None
        self._data: xr.Dataset | None = None
        self._load_all()

    def _load_all(self) -> None:
        first_path = self.paths[0]
        ds = xr.open_dataset(first_path)
        has_hp = "cluster" in ds and "hp_pixel" in ds["cluster"].dims
        ds.close()

        if has_hp:
            if self.nside is None:
                raise ValueError(
                    "First file has HealPix format. Pass nside=... for MMA."
                )
            for path in self.paths:
                ds = xr.open_dataset(path)
                file_nside = ds.attrs.get("nside")
                ds.close()
                if file_nside is not None and int(file_nside) != self.nside:
                    raise ValueError(
                        f"File {path} has nside={file_nside}, but MMA uses nside={self.nside}. "
                        "All files must use the same nside."
                    )
                masks = _load_ever_in_healpix_masks(path, self.nside)
                self._cluster_masks.append(masks)
            self._format = "healpix"
            logger.info(
                f"MMA: loaded {len(self._cluster_masks)} HealPix cluster file(s) (ever-in), nside={self.nside}"
            )
        else:
            if self.nside is not None:
                raise ValueError(
                    "First file has native format. Use nside=None for MMA with native exports."
                )
            ref_shape: Optional[tuple[int, ...]] = None
            for path in self.paths:
                masks, coords_2d, use_sphere, spatial_dims, spatial_shape = (
                    _load_ever_in_native_masks(path)
                )
                if ref_shape is None:
                    ref_shape = spatial_shape
                    self._native_coords_2d = coords_2d
                    self._native_use_sphere = use_sphere
                    self._native_spatial_dims = spatial_dims
                    self._native_spatial_shape = spatial_shape
                elif spatial_shape != ref_shape:
                    raise ValueError(
                        f"File {path} has spatial shape {spatial_shape}, "
                        f"expected {ref_shape}. All native files must have the same grid."
                    )
                self._cluster_masks.append(masks)
            self._format = "native"
            logger.info(
                f"MMA: loaded {len(self._cluster_masks)} native cluster file(s) (ever-in)"
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
            k_neighbors: K for KNN graph.
            top_n_clusters: If set, only consider top N clusters per model.
            show_progress: Whether to show progress.

        Returns:
            The internal dataset with consensus_clusters and consensus_consistency.
        """
        shared_attrs = {
            "min_consensus": min_consensus,
            "min_cluster_size": min_cluster_size,
            "top_n_clusters": top_n_clusters,
            "k_neighbors": k_neighbors,
            "n_models": len(self._cluster_masks),
            "source_paths": self.paths,
            _attrs.METHOD_NAME: "mma_consensus",
            _attrs.TOAD_VERSION: __version__,
        }

        if self._format == "healpix":
            labels, consistency = run_healpix_consensus(
                self._cluster_masks,
                cast(int, self.nside),
                min_consensus=min_consensus,
                min_cluster_size=min_cluster_size,
                k_neighbors=k_neighbors,
                top_n_clusters=top_n_clusters,
                show_progress=show_progress,
            )
            npix = 12 * cast(int, self.nside) ** 2
            da_clusters = xr.DataArray(
                labels,
                dims=["hp_pixel"],
                coords={"hp_pixel": np.arange(npix)},
                name="consensus_clusters",
                attrs={
                    **shared_attrs,
                    "nside": self.nside,
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
                    "nside": self.nside,
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
        else:
            n_spatial = cast(np.ndarray, self._native_coords_2d).shape[0]
            knn_rows, knn_cols = _build_knn_edges_from_coords_2d(
                cast(np.ndarray, self._native_coords_2d),
                k=k_neighbors,
                use_sphere=self._native_use_sphere,
            )
            labels, consistency = run_native_consensus(
                self._cluster_masks,
                n_spatial=n_spatial,
                knn_rows=knn_rows,
                knn_cols=knn_cols,
                min_consensus=min_consensus,
                min_cluster_size=min_cluster_size,
                top_n_clusters=top_n_clusters,
                show_progress=show_progress,
            )
            spatial_dims = cast(List[str], self._native_spatial_dims)
            spatial_shape = cast(tuple[int, ...], self._native_spatial_shape)
            labels_2d = labels.reshape(spatial_shape)
            consistency_2d = consistency.reshape(spatial_shape)
            # Get coords from first export
            ds0 = xr.open_dataset(self.paths[0])
            coords = {d: ds0[d] for d in spatial_dims if d in ds0}
            ds0.close()
            da_clusters = xr.DataArray(
                labels_2d,
                dims=spatial_dims,
                coords=coords,
                name="consensus_clusters",
                attrs={
                    **shared_attrs,
                    "description": "Spatial consensus clusters on native grid (time-collapsed).",
                    _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CLUSTER,
                    _attrs.CONSENSUS_CONSISTENCY_VARIABLE: "consensus_consistency",
                },
            )
            da_consistency = xr.DataArray(
                consistency_2d,
                dims=spatial_dims,
                coords=coords,
                name="consensus_consistency",
                attrs={
                    **shared_attrs,
                    "description": "Consistency scores for each native grid cell.",
                    _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CONSISTENCY,
                },
            )
            self.data = xr.merge(
                [da_clusters, da_consistency],
                combine_attrs="override",
                compat="override",
            )
        self.data.attrs["format"] = self._format

        labels_flat = self.data["consensus_clusters"].values.ravel()
        n_clusters = len(
            np.unique(labels_flat[(labels_flat >= 0) & np.isfinite(labels_flat)])
        )
        logger.info(
            f"MMA consensus: {n_clusters} clusters, "
            f"noise={int(np.sum(labels_flat == -1))}, "
            f"nan={int(np.sum(np.isnan(labels_flat)))}"
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

    def get_healpix_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Get (lat, lon) in degrees for each HealPix pixel, for plotting with cartopy etc.

        Returns:
            lats, lons: 1D arrays of length npix (12 * nside**2).

        Raises:
            AttributeError: If run_consensus() has not been called yet.
            ValueError: If MMA uses native format (use data coords instead).
        """
        if self._format != "healpix":
            raise ValueError(
                "get_healpix_latlon is for HealPix format. "
                "For native format, use data['consensus_clusters'] coords."
            )
        regridder = HealPixRegridder(nside=cast(int, self.nside))
        npix = 12 * cast(int, self.nside) ** 2
        return regridder.pixels_to_latlon(np.arange(npix))

    def map_consensus_to_coords(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Assign consensus cluster ID to each (lat, lon) by HealPix index lookup.

        HealPix format only. For native format, use map_consensus_to_dataset.

        Args:
            lat: Latitude(s) in degrees. Any shape.
            lon: Longitude(s) in degrees. Same shape as lat.

        Returns:
            Array of same shape as lat/lon with consensus cluster IDs.
        """
        if self._format != "healpix":
            raise ValueError(
                "map_consensus_to_coords is for HealPix format. "
                "For native format, use map_consensus_to_dataset."
            )
        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have the same shape")
        regridder = HealPixRegridder(nside=cast(int, self.nside))
        coords_2d = np.column_stack([lat.ravel(), lon.ravel()])
        hp_indices = regridder.map_orig_to_regrid(coords_2d)
        clusters = self.data["consensus_clusters"].values
        return clusters[hp_indices].reshape(lat.shape)

    def map_consensus_to_dataset(self, ds: xr.Dataset) -> xr.DataArray:
        """Assign consensus cluster IDs to an xarray Dataset's spatial grid.

        Uses lat/lon (HealPix) or same grid (native) from the dataset.

        Args:
            ds: Dataset with lat/lon (HealPix) or matching spatial dims (native).

        Returns:
            DataArray of consensus cluster IDs with same spatial dims as ds.
        """
        if self._format == "healpix":
            lat_name, lon_name = detect_latlon_names(ds)
            if lat_name is None or lon_name is None:
                raise ValueError(
                    "Dataset must have lat/lon coordinates for HealPix MMA lookup."
                )
            lat = ds[lat_name].values
            lon = ds[lon_name].values
            if lat.ndim == 1 and lon.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
                dims = list(ds[lat_name].dims) + list(ds[lon_name].dims)
            else:
                dims = list(ds[lat_name].dims)
            cluster_ids = self.map_consensus_to_coords(lat, lon)
        else:
            cc = self.data["consensus_clusters"]
            for d in cc.dims:
                if d not in ds:
                    raise ValueError(
                        f"Dataset must have dimension '{d}' for native MMA (same grid)."
                    )
            cluster_ids = np.asarray(cc.values)
            dims = list(cc.dims)
        coords = {d: ds[d] for d in dims if d in ds}
        return xr.DataArray(
            cluster_ids,
            dims=dims,
            coords=coords,
            name="consensus_cluster_id",
            attrs={"description": "Consensus cluster ID per grid cell"},
        )

    def get_shift_times_from_export(
        self,
        export_path: str,
        consensus_cluster_id: int,
        numeric: bool = True,
    ) -> np.ndarray:
        """Extract shift times for a consensus cluster from an MMA export file.

        The export must contain the ``cluster`` variable (time × space), written
        when using ``export_for_mma``. Space is either hp_pixel (HealPix) or native dims.

        Args:
            export_path: Path to the per-model export file (from compute_clusters).
            consensus_cluster_id: Consensus cluster ID (e.g. 0, 1, 2).
            numeric: If True, return numeric time values. If False, return native times.

        Returns:
            Flattened array of time values for events in the consensus region.
        """
        ds = xr.open_dataset(export_path)
        if "cluster" not in ds:
            ds.close()
            raise ValueError(
                f"Export file {export_path} has no 'cluster' variable. "
                "Re-export with compute_clusters(export_for_mma=...) to include it."
            )
        clusters = ds["cluster"]
        time_dims = [d for d in clusters.dims if d in ds.coords]
        if not time_dims:
            ds.close()
            raise ValueError(f"Export {export_path} cluster has no time dimension.")
        time_dim = time_dims[0]
        time_coord = ds[time_dim]

        if "hp_pixel" in clusters.dims:
            # HealPix format: lookup in HealPix space, no over-counting
            consensus = self.data["consensus_clusters"].values
            region_mask = (consensus == consensus_cluster_id) & np.isfinite(consensus)
            arr = clusters.values
            in_cluster = (arr != -1) & np.isfinite(arr)
            combined = region_mask[np.newaxis, :] & in_cluster
            event_times = np.broadcast_to(time_coord.values[:, np.newaxis], arr.shape)[
                combined
            ]
            ds.close()
        else:
            # Native format: map consensus to dataset grid
            consensus_ids = self.map_consensus_to_dataset(ds)
            region_mask = (
                consensus_ids == consensus_cluster_id
            ) & consensus_ids.notnull()
            in_cluster = (clusters != -1) & clusters.notnull()
            combined = region_mask.broadcast_like(clusters) & in_cluster
            t = xr.DataArray(
                time_coord.values,
                dims=[time_dim],
                coords={time_dim: time_coord},
            )
            t_broadcast = t.broadcast_like(clusters)
            event_times = t_broadcast.where(combined).values.ravel()
            ds.close()

        if numeric:
            out = event_times[np.isfinite(event_times)]
        else:
            out = event_times[event_times == event_times]
        return np.asarray(out)

    def get_shift_times_per_consensus_cluster(
        self,
        numeric: bool = True,
    ) -> Dict[int, np.ndarray]:
        """Aggregate shift times for each consensus cluster across all input exports.

        For each consensus cluster ID, collects shift times from every per-model
        export file and returns them as a single array per cluster.

        Returns:
            Dict mapping consensus cluster ID to 1D array of shift times
            (aggregated across all exports).

        Example:
            >>> times_by_cluster = mma.get_shift_times_per_consensus_cluster()
            >>> for cid, times in times_by_cluster.items():
            ...     plt.hist(times, bins=30, alpha=0.5, label=f"Cluster {cid}")
        """
        clusters = self.data["consensus_clusters"].values
        ids = np.unique(clusters[(clusters >= 0) & np.isfinite(clusters)])
        ids = [int(x) for x in ids]

        out: Dict[int, np.ndarray] = {}
        for cid in ids:
            times_list: List[np.ndarray] = []
            for path in self.paths:
                times_list.append(
                    self.get_shift_times_from_export(
                        path,
                        consensus_cluster_id=cid,
                        numeric=numeric,
                    )
                )
            out[cid] = np.concatenate(times_list) if times_list else np.array([])
        return out

    def plot_consensus_clusters(
        self,
        ax: Optional[Axes] = None,
        map_style: Optional[Union[Dict[str, Any], "MapStyle"]] = None,
        cmap: str = "tab10",
        s: float = 10,
        vmin: float | None = None,
        vmax: float | None = None,
        show_noise: bool = False,
        add_colorbar: bool = True,
        cluster_ids: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> Tuple[Figure, Axes]:
        """Plot consensus clusters on a map.

        Args:
            ax: Axes to plot on. If None, creates a new figure with map features.
            map_style: Map style (projection, coastlines, etc.). Pass a dict, e.g.
                ``{"projection": ccrs.Orthographic(-40, 15), "continent_shading": True}``
                or ``{"projection": "mollweide"}``. Used only when ax is None.
            cmap: Colormap for clusters. Defaults to "tab10".
            s: Scatter point size. Defaults to 10.
            vmin: Lower bound for colour scale. Defaults to -0.5.
            vmax: Upper bound for colour scale. Defaults to 9.5.
            show_noise: If True, plot noise pixels (-1) as grey. Defaults to False.
            add_colorbar: Whether to add a colourbar. Defaults to True.
            cluster_ids: Optional sequence of cluster IDs to plot. If specified, only these clusters will be shown.
            **kwargs: Passed through to ``ax.scatter``.

        Returns:
            Tuple of (figure, axes).

        Raises:
            AttributeError: If run_consensus() has not been called yet.
            ValueError: If MMA uses native format (use data['consensus_clusters'].plot()).
        """
        if self._format != "healpix":
            raise ValueError(
                "plot_consensus_clusters is for HealPix format (lat/lon maps). "
                "For native format, use data['consensus_clusters'].plot() or similar."
            )
        from toad.plotting import (
            _add_map_features,
            _normalize_map_style,
            get_projection,
        )

        clusters = self.data["consensus_clusters"].values
        lats, lons = self.get_healpix_latlon()

        # Mask for clusters to plot: >=0, finite, and in cluster_ids if specified.
        cluster_mask = (clusters >= 0) & np.isfinite(clusters)
        if cluster_ids is not None:
            cluster_ids_set = set(cluster_ids)
            cluster_mask &= np.isin(clusters, list(cluster_ids_set))
        noise_mask = clusters == -1

        if ax is None:
            config = _normalize_map_style(map_style)
            proj = (
                get_projection(config.projection)
                if config.projection is not None
                else ccrs.Mollweide()
            )
            fig, ax = plt.subplots(subplot_kw=dict(projection=proj))
            _add_map_features(cast(GeoAxes, ax), config)
            cast(GeoAxes, ax).set_global()
        else:
            fig = cast(Figure, ax.get_figure())

        if show_noise:
            ax.scatter(
                lons[noise_mask],
                lats[noise_mask],
                c="lightgray",
                s=s,
                transform=ccrs.PlateCarree(),
                zorder=0,
                **kwargs,
            )

        sc = ax.scatter(
            lons[cluster_mask],
            lats[cluster_mask],
            c=clusters[cluster_mask],
            s=s,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            **kwargs,
        )

        if add_colorbar:
            plt.colorbar(sc, ax=ax, label="Cluster ID")

        return cast(Tuple[Figure, Axes], (fig, ax))
