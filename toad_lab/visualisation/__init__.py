import xarray as xr
import numpy as np
from pyproj import Proj, transform
# from plotting_functions import *
# from pyunicorn.timeseries import RecurrenceNetwork, RecurrencePlot
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from toad_lab import TOAD
import cartopy.crs as ccrs

class TOADPlotter:
    def __init__(self, td: TOAD):
        """
        Initialize the plotter with a jTOAD object.
        """
        self.td = td

    def plot_clusters_on_map(self, var, cluster_label=None, cluster_ids=None, ax=None, cmap="tab20", time_dim="time"):
        """
        Plot the clusters on a South Pole map.
        Paramss: 
        -   clusters: supply your own cluster labels (e.g. from a different clustering algorithm)
        """
        if cluster_label is None:
            cluster_label = self.td.get_clusters(var)

        if ax is None:
            fig, ax = plt.subplots()

        if cluster_ids is None:
            cluster_ids = np.unique(cluster_label)
            cluster_ids = cluster_ids[cluster_ids != -1]
        
        im = cluster_label.where(cluster_label.isin(cluster_ids)).max(dim=time_dim).plot(ax=ax, cmap=cmap, add_colorbar=False)

        # add_colorbar(ax, im, 'Cluster IDs')
        ax.set_title(f'{var}_cluster')
        return self

    def plot_cluster_on_map(self, var, cluster_id, color="k", ax=None, time_dim="time"):
        """
        Plot the clusters on a South Pole map.
        """
        if ax is None:
            fig, ax = plt.subplots()

        clusters = self.td.get_clusters(var)
        data_mask = self.td.data[var].max(dim=time_dim) > 0
        clusters.where(data_mask).where(clusters.max(dim=time_dim) == cluster_id).max(dim=time_dim).plot(ax=ax, cmap=ListedColormap([color]), add_colorbar=False)
        ax.set_title(f'{var}_cluster {cluster_id}')
        return self


    def plot_clusters_on_maps(self, var, max_clusters = 5, ncols = 5, color="k", south_pole=False):
        cluster_counts = self.td.get_cluster_counts(var)
        n_clusters = np.min([len(self.td.get_clusters(var).clusters), max_clusters])
        nrows = int(np.ceil(n_clusters / ncols))
        # fig, axs = south_pole_plots(nrows, ncols, h=nrows*3)
        # Intsead of south_pole_plots:

        projection = ccrs.SouthPolarStereo() if south_pole else ccrs.PlateCarree()
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows*2.5), subplot_kw={'projection': projection})
        for ax in axs.flat:
            ax.coastlines(resolution="110m")
            if(south_pole):
                ax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
        
        for i, id in enumerate(self.td.get_largest_cluster_ids(var)[:n_clusters]):
            ax = axs.flat[i]
            self.plot_cluster_on_map(var, ax=ax, cluster_id=id, color=color)
            ax.set_title(f"id {id} with {cluster_counts[id]} members", fontsize=10)


    def plot_shifts(self, var, frame=0, ax=None, cmap="RdBu"):
        """
        Plot the shifts (rate of change and ice thickness) for a specific time frame.
        """

        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot rate of change
        self.td.data[f"{var}_dts"].isel(time=frame).plot(cmap=cmap, ax=ax)
        ax.set_title(f'{var}_dts')
        return self

    
    def plot_var(self, var, frame=0, ax=None, cmap="RdBu", **kwargs):
        """
        Plot the shifts (rate of change and ice thickness) for a specific time frame.
        """

        if ax is None:
            fig, ax = plt.subplots()
        
        # Plot rate of change
        self.td.data[var].isel(time=frame).plot(cmap=cmap, ax=ax, **kwargs)
        ax.set_title(var)
        return self

    def plot_cluster_time_series(self, var, cluster_id, ax=None, max_trajectories=1_000, plot_shifts=False, **plot_kwargs):
        timeseries = self.td.get_timeseries_in_cluster(var, cluster_id)
        if(plot_shifts):
            timeseries = [ts[var + "_dts"] for ts in timeseries]
        else:
            timeseries = [ts[var] for ts in timeseries]
        
        if ax is None:
            fig, ax = plt.subplots()


        # Limit the number of trajectories to plot
        max_trajectories = np.min([max_trajectories, len(timeseries)])

        # Shuffle the timeseries to get a random sample
        order = np.arange(len(timeseries))
        np.random.shuffle(order) 
        order = order[:max_trajectories]

        for i in order:
            timeseries[i].plot(ax=ax, **plot_kwargs)
        
        if max_trajectories < len(timeseries):
            ax.set_title(f'Random sample of {max_trajectories} from total {len(timeseries)} timeseries for {var} in cluster {cluster_id}')
        else:                                                                              
            ax.set_title(f'{len(timeseries)} timeseries for {var} in cluster {cluster_id}')
        return self


    # ============================================================
    #               Cluster map + time series plot
    # ============================================================

    # def cluster_view_timeseries_and_map(self, var, cluster_id, south_pole=False):
    #     fig, axs = plots(2,1, squeeze=False, h_ratios=[4,1], h=PLT_HEIGHT_HALF*0.8, w=PLT_WIDTH_FULL)
    #     if south_pole:
    #         replace_ax_south_pole(fig, axs, 0, 0, linewidth=0.5)
    #     else:
    #         replace_ax_map(fig, axs, 0, 0, linewidth=0.5)
    #     axs = axs.flat
    #     self.plot_cluster_on_map(var, ax=axs[0], cluster_id=cluster_id, color="plum")
    #     self.plot_cluster_time_series(var, ax=axs[1], cluster_id=cluster_id, color="plum", alpha=0.5, lw=0.5, max_trajectories=500)
    #     cluster_persistence_fraction = self.td.cluster_persistence_fraction(var, cluster_id)
    #     span = non_zero_span(cluster_persistence_fraction)
    #     time = self.td.data.time
    #     axs[1].axvspan(time[span[0]], time[span[1]], color='blue', alpha=0.1)
    #     axs0_twin = axs[1].twinx()
    #     axs0_twin.set_ylim(-0.05, 1.05)
    #     axs0_twin.set_ylabel("Cluster persistence fraction", color=C1)
    #     axs0_twin.plot(self.td.data.time, cluster_persistence_fraction)
        
    #     for ax in [*axs, axs0_twin]:
    #         ax.set_title(ax.get_title(), fontsize=PLT_FONT_SIZE_08)
    #         ax.set_ylabel(ax.get_ylabel(), fontsize=PLT_FONT_SIZE_08)
    #         ax.set_xlabel(ax.get_xlabel(), fontsize=PLT_FONT_SIZE_08)


    # def cluster_view_timeseries_and_map_interactive(self, var, south_pole=False):
    #     from ipywidgets import interact
    #     largest_cluster_idx = self.td.get_n_largest_cluster_ids(var, n=10)
    #     @interact(i=(0, len(largest_cluster_idx) - 1))
    #     def _(i=1): self.cluster_view_timeseries_and_map(var, largest_cluster_idx[i], south_pole=south_pole)


    # # With dts plot
    # def cluster_view_timeseries_and_map2(self, var, cluster_id, south_pole=False):
    #     fig, axs = plots(3,1, squeeze=False, h_ratios=[4,1,1], h=PLT_HEIGHT_HALF*0.9, w=PLT_WIDTH_FULL)
    #     if south_pole:
    #         replace_ax_south_pole(fig, axs, 0, 0, linewidth=0.5)
    #     else:
    #         replace_ax_map(fig, axs, 0, 0, linewidth=0.5)
    #     axs = axs.flat
    #     self.plot_cluster_on_map(var, ax=axs[0], cluster_id=cluster_id, color="plum")
    #     self.plot_cluster_time_series(var, ax=axs[1], cluster_id=cluster_id, color="plum", alpha=0.5, lw=0.5, max_trajectories=500)
    #     self.plot_cluster_time_series(var, ax=axs[2], cluster_id=cluster_id, color="plum", alpha=0.5, lw=0.5, max_trajectories=500, plot_shifts=True)

    #     axs[1].sharex(axs[2])

    #     cluster_persistence_fraction = self.td.cluster_persistence_fraction(var, cluster_id)
    #     span = non_zero_span(cluster_persistence_fraction)
    #     time = self.td.data.time
    #     axs[1].axvspan(time[span[0]], time[span[1]], color='blue', alpha=0.1)
    #     axs0_twin = axs[1].twinx()
    #     axs0_twin.set_ylim(-0.05, 1.05)
    #     axs0_twin.set_ylabel("Cluster persistence fraction", color=C1)
    #     axs0_twin.plot(self.td.data.time, cluster_persistence_fraction)
        
    #     for ax in [*axs, axs0_twin]:
    #         ax.set_title(ax.get_title(), fontsize=PLT_FONT_SIZE_08)
    #         ax.set_ylabel(ax.get_ylabel(), fontsize=PLT_FONT_SIZE_08)
    #         ax.set_xlabel(ax.get_xlabel(), fontsize=PLT_FONT_SIZE_08)
        
    #     axs[1].set_xlabel("")



    # def cluster_view_timeseries_and_map_interactive2(self, var, south_pole=False):
    #     from ipywidgets import interact
    #     largest_cluster_idx = self.td.get_n_largest_cluster_ids(var, n=10)
    #     @interact(i=(0, len(largest_cluster_idx) - 1))
    #     def _(i=1): self.cluster_view_timeseries_and_map2(var, largest_cluster_idx[i], south_pole=south_pole)

