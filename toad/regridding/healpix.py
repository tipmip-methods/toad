import numpy as np
import healpix as hp
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional

from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs

from toad.regridding.base import BaseRegridder


import logging

logger = logging.getLogger("TOAD")


class HealPixRegridder(BaseRegridder):
    """Regrid data onto a equal-area HEALPix grid to avoid polar bias in clustering"""

    def __init__(self, nside: Optional[int] = None):
        """
        Args:
            nside: HEALPix parameter nside, which must be a power of 2. The total number of pixels in the regridded grid (npix) is calculated using the formula: npix = 12 * nside ** 2. If nside is not specified, it will be automatically determined based on the data's resolution.
        """

        self.df_healpix: pd.DataFrame = pd.DataFrame()

        # Make sure nside is a power of 2
        if nside is not None and not np.log2(nside).is_integer():
            raise ValueError(f"nside must be a power of 2, got {nside}")
        self.nside = nside

    def latlon_to_healpix(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Convert arrays of latitude and longitude to HEALPix pixel indices."""
        return hp.ang2pix(self.nside, lons, lats, lonlat=True)

    def healpix_to_latlon(self, pix: int) -> tuple:
        """Convert a HEALPix pixel index back to its center latitude and longitude."""
        theta, phi = hp.pix2ang(self.nside, pix)
        return 90 - np.degrees(theta), np.degrees(phi)  # lat, lon

    def regrid(
        self, coords: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Regrid data to new coordinate system.

        Args:
            coords: 3dArray of coordinates (time, lat, lon) in that order
            weights: 1dArray of weights
            nside: Optional[int] = None, if provided, use this nside for regridding
        Returns:
            3dArray of coordinates (time, lat, lon) in that order
            1dArray of weights
        """
        # Store original spatial coordinates
        self.original_coords = coords
        self.original_weights = weights

        # If nside is not provided, compute it automatically based on the resolution of the data
        if self.nside is None:
            n_pixels = len(np.unique(coords[:, 1])) * len(
                np.unique(coords[:, 2])
            )  # this assumes that the original grid is a regular grid..
            order = 0.5 * np.log2(n_pixels / 12.0)      # this and next line is implementation 
            self.nside = 1 << int(np.ceil(order))       # of healpy.pixelfunc.get_min_valid_nside(n_pixels)
            logger.info(f"Automatically computed nside: {self.nside}")

        # Get unique lat/lon pairs and compute healpix indices once
        unique_coords = np.unique(coords[:, 1:], axis=0)  # unique lat/lon pairs
        unique_hp_indices = self.latlon_to_healpix(
            unique_coords[:, 0], unique_coords[:, 1]
        )

        # Create mapping from lat/lon to healpix index
        coord_to_hp = {
            (lat, lon): hp_idx
            for (lat, lon), hp_idx in zip(map(tuple, unique_coords), unique_hp_indices)
        }

        # Create DataFrame with mapped healpix indices
        df = pd.DataFrame(
            {
                "time": coords[:, 0],
                "lat": coords[:, 1],
                "lon": coords[:, 2],
                "vals": weights,
                "hp_pix": [
                    coord_to_hp[(lat, lon)]
                    for lat, lon in zip(coords[:, 1], coords[:, 2])
                ],
            }
        )

        # Group and aggregate
        group_cols = [
            "time",
            "hp_pix",
        ]  # This means if multiple points fall in the same HEALPix pixel at the same time, they get averaged.
        df = df.groupby(group_cols)["vals"].mean().reset_index()

        # Add regridded coordinates
        df["lat"], df["lon"] = zip(
            *df["hp_pix"].apply(self.healpix_to_latlon)
        )  # Convert healpix index back to lat, lon

        if np.any(np.isnan(df["hp_pix"])):
            logger.warning(
                "Warning: Interpolation contains NaNs. Consider decreasing nside_p"
            )

        self.df_healpix = df
        return np.column_stack([df["time"], df["lat"], df["lon"]]), df[
            "vals"
        ].to_numpy()

    def regrid_clusters_back(self, cluster_labels: np.ndarray) -> np.ndarray:
        """Map cluster labels back to original grid."""
        if self.original_coords is None or self.df_healpix is None:
            raise ValueError("Must call regrid() first")

        # Add cluster labels to healpix DataFrame
        self.df_healpix["cluster"] = cluster_labels

        # Create mapping dictionary from (time, hp_pix) to cluster label
        mapping = dict(
            zip(
                zip(
                    self.df_healpix["time"], self.df_healpix["hp_pix"]
                ),  # group by both time and hp_pix because hp_pix is constant with time
                cluster_labels,
            )
        )

        # Calculate healpix indices for original points
        hp_indices = self.latlon_to_healpix(
            self.original_coords[:, 1], self.original_coords[:, 2]
        )

        # Map back using time and healpix indices
        result = np.array(
            [
                mapping.get((time, hp), -1)  # if no mapping, return -1
                for time, hp in zip(self.original_coords[:, 0], hp_indices)
            ]
        )

        return result

    def plot(
        self, val_var: str = "cluster", time=None, cmap="coolwarm", center_lon=180
    ):
        """Plot regridded data in HEALPix projection."""
        raise NotImplementedError("Demo plot not implemented yet. Missing healpy.mollview surrogate.")
        """
        if self.df_healpix is None:
            raise ValueError("No data available. Run regrid() first")
        df = self.df_healpix

        plot_df = df[df["time"] == time] if time is not None else df

        sparse_map = np.zeros(hp.nside2npix(self.nside))
        sparse_map[plot_df["hp_pix"]] = plot_df[val_var]

        # no surrogate for healpy.mollview function found yet
        healpy.mollview(
            sparse_map,
            title=f"HEALPix Grid (nside={self.nside})"
            + (f" at {time}" if time else ""),
            cmap=cmap,
            unit=val_var,
            rot=(center_lon, 0, 0),
        )
        plt.show()"""

    def plot_clusters(
        self, s=1, cmap=None, color=None, ax=None, extent=None, add_colorbar=True
    ):
        #raise NotImplementedError()
        if ax is None:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection=ccrs.Mollweide())
            ax.coastlines()  # type: ignore

        if self.df_healpix is None:
            raise ValueError("No data available. Run regrid() first")

        df_plot = self.df_healpix[self.df_healpix["cluster"] >= 0]

        if cmap is None:
            N_clusters = len(df_plot["cluster"].unique())
            cmap = (
                "tab10" if N_clusters <= 10 else "tab20" if N_clusters <= 20 else "jet"
            )

        if color is not None:
            cmap = ListedColormap([color])

        sc = ax.scatter(
            df_plot["lon"],
            df_plot["lat"],
            c=df_plot["cluster"],
            s=s,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
        )
        if add_colorbar:
            plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05)
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())  # type: ignore

    def demo_plot(self):
        """Demo the HEALPix grid with a simple latitude-based pattern"""

        raise NotImplementedError("Demo plot not implemented yet. Missing healpy.mollview surrogate.")
        """
        if self.nside is None:
            raise ValueError(
                "Please provide an nside value in the HealPixRegridder constructor for the demo plot."
            )

        # Generate evenly spaced points across the sphere
        npix = hp.nside2npix(self.nside)
        pixels = np.arange(npix)

        # Get coordinates in lat/lon
        lons, lats = hp.pix2ang(self.nside, pixels, lonlat=True)

        # Create demo data
        time_dummy = np.zeros(npix)
        vals = lats

        # Run through regridding pipeline
        coords = np.column_stack([time_dummy, lats, lons])
        coords, weights = self.regrid(coords, vals)

        # Plot results
        self.plot(val_var="vals", cmap="RdBu_r")"""
