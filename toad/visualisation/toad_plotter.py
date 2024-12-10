import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.figure


class TOADPlotter:
    """
    Plotting methods for TOAD objects.
    
    Note: Docstrings here are short as this class is under heavy development
    """
    
    def __init__(self, td):
        """Init TOADPlotter with a TOAD object """
        self.td = td


<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
    # TODO make function for contour plot: td.get_spatial_cluster_mask("thk", id).plot.contour(levels=1)
    # TODO make function for plotting snap shots of cluster
        # start, end = td.cluster_stats("thk").time.start(id), td.cluster_stats("thk").time.end(id)
        # td.apply_cluster_mask("thk", "thk", cluster_id).sel(time=slice(start, end, 5)).plot(col='time', col_wrap=5, cmap='jet')


    def map_plots(self, nrows=1, ncols=1, projection=ccrs.PlateCarree(), resolution="110m", linewidth=(0.5, 0.25), grid_labels=True, grid_style='--', grid_width=0.5, grid_color='gray', grid_alpha=0.5, figsize=None, borders=True, **kwargs) -> tuple[matplotlib.figure.Figure, np.ndarray]:
<<<<<<< HEAD
=======
    def map_plots(self, nrows=1, ncols=1, projection=ccrs.PlateCarree(), resolution="110m", linewidth=(0.5, 0.25), grid_labels=True, grid_style='--', grid_width=0.5, grid_color='gray', grid_alpha=0.5, figsize=None, borders=True, **kwargs):
>>>>>>> c6fc662 (Docstring and type fixes)
=======
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
        """
        Plot maps with coastlines, gridlines, and optional borders.
        """

        fig = plt.figure(figsize=figsize)
        if nrows == 1 and ncols == 1:
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            axs = ax
        else:
            axs = np.empty((nrows, ncols), dtype=object)
            for i in range(nrows):
                for j in range(ncols):
                    axs[i,j] = fig.add_subplot(nrows, ncols, i*ncols + j + 1, projection=projection)
                    
        # Add features to all axes
        if isinstance(axs, np.ndarray):
            for ax in axs.flat:
                ax.coastlines(resolution=resolution, linewidth=linewidth[0])
                if borders:
                    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=linewidth[1])
                if grid_labels:
                    ax.gridlines(draw_labels=grid_labels, linewidth=grid_width, color=grid_color, alpha=grid_alpha, linestyle=grid_style)
        else:
            axs.coastlines(resolution=resolution, linewidth=linewidth[0]) # type: ignore
            if borders:
                axs.add_feature(cfeature.BORDERS, linestyle='-', linewidth=linewidth[1]) # type: ignore
            if grid_labels:
                axs.gridlines(draw_labels=grid_labels, linewidth=grid_width, color=grid_color, alpha=grid_alpha, linestyle=grid_style) # type: ignore
                
        # TODO fix type error
        return fig, axs



    def south_pole_plots(self, nrows=1, ncols=1, resolution="110m", linewidth=(0.5, 0.25), grid_labels=True, grid_style='--', grid_width=0.5, grid_color='gray', grid_alpha=0.5, figsize=None, borders=True, **kwargs):
        """
        Plot maps with coastlines, gridlines, and optional borders at the South Pole.
        """
        fig, axs = self.map_plots(nrows, ncols, projection=ccrs.SouthPolarStereo(), resolution=resolution, linewidth=linewidth, grid_labels=grid_labels, grid_style=grid_style, grid_width=grid_width, grid_color=grid_color, grid_alpha=grid_alpha, figsize=figsize, borders=borders, **kwargs)
        if isinstance(axs, np.ndarray):
            axs_flat = axs.flat
        else:
            axs_flat = [axs]

        for ax in axs_flat:
<<<<<<< HEAD
<<<<<<< HEAD
            ax.coastlines(resolution="110m", linewidth=linewidth[0]) # type: ignore
=======
            ax.coastlines(resolution="110m") # type: ignore
>>>>>>> c6fc662 (Docstring and type fixes)
=======
            ax.coastlines(resolution="110m", linewidth=linewidth[0]) # type: ignore
>>>>>>> efd56b8 (Fix plotting clusters)
            ax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree()) # type: ignore
        return fig, axs

<<<<<<< HEAD
<<<<<<< HEAD
    def plot_clusters_on_map(self, var, cluster_ids=None, ax=None, cmap="tab20"):
=======
    def plot_clusters_on_map(self, var, cluster_ids=None, ax=None, cmap="tab20", time_dim="time"):
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)
=======
    def plot_clusters_on_map(self, var, cluster_ids=None, ax=None, cmap="tab20"):
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
        """
        Plot the clusters on a map.
        
<<<<<<< HEAD
        >> Args: 
            var:
                name of the variable for which clusters have been computed or the name of the custom cluster variable.
            cluster_ids:
                which clusters to plot, defaults to all clusters
=======
        Args: 
            - var: name of the variable for which clusters have been computed or the name of the custom cluster variable.
            - cluster_ids: which clusters to plot, defaults to all clusters
>>>>>>> c6fc662 (Docstring and type fixes)
        """
        clusters = self.td.get_clusters(var)

        if ax is None:
            fig, ax = plt.subplots()

        if cluster_ids is None:
            cluster_ids = np.unique(clusters)
            cluster_ids = cluster_ids[cluster_ids != -1]
        
<<<<<<< HEAD
<<<<<<< HEAD
        im = clusters.where(clusters.isin(cluster_ids)).max(dim=self.td.time_dim).plot(ax=ax, cmap=cmap, add_colorbar=False)
=======
        im = clusters.where(clusters.isin(cluster_ids)).max(dim=time_dim).plot(ax=ax, cmap=cmap, add_colorbar=False)
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)
=======
        im = clusters.where(clusters.isin(cluster_ids)).max(dim=self.td.time_dim).plot(ax=ax, cmap=cmap, add_colorbar=False)
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)

        # add_colorbar(ax, im, 'Cluster IDs')
        ax.set_title(f'{clusters.name}')
        return self

    def plot_cluster_on_map(self, var, cluster_id, color="k", ax=None):
        """
        Plot a individual clusters on a map.
        """
        if ax is None:
            fig, ax = plt.subplots()

        clusters = self.td.get_clusters(var)
<<<<<<< HEAD
<<<<<<< HEAD
        data_mask = self.td.data[var].max(dim=self.td.time_dim) > 0
        if cluster_id == -1:
            # Completely un-clustered cells are those that never have a cluster_id higher than -1
            clusters.where(data_mask).where(clusters.max(dim=self.td.time_dim) == cluster_id).max(dim=self.td.time_dim).plot(ax=ax, cmap=ListedColormap([color]), add_colorbar=False)
        else:
            clusters.where(data_mask).where(clusters == cluster_id).max(dim=self.td.time_dim).plot(ax=ax, cmap=ListedColormap([color]), add_colorbar=False)
=======
        data_mask = self.td.data[var].max(dim=time_dim) > 0
=======
        data_mask = self.td.data[var].max(dim=self.td.time_dim) > 0
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
        if cluster_id == -1:
            # Completely un-clustered cells are those that never have a cluster_id higher than -1
            clusters.where(data_mask).where(clusters.max(dim=self.td.time_dim) == cluster_id).max(dim=self.td.time_dim).plot(ax=ax, cmap=ListedColormap([color]), add_colorbar=False)
        else:
<<<<<<< HEAD
            clusters.where(data_mask).where(clusters == cluster_id).max(dim=time_dim).plot(ax=ax, cmap=ListedColormap([color]), add_colorbar=False)
>>>>>>> efd56b8 (Fix plotting clusters)
=======
            clusters.where(data_mask).where(clusters == cluster_id).max(dim=self.td.time_dim).plot(ax=ax, cmap=ListedColormap([color]), add_colorbar=False)
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
        ax.set_title(f'{var}_cluster {cluster_id}')
        return self


    def plot_clusters_on_maps(self, var, max_clusters = 5, ncols = 5, color="k", south_pole=False):
        """
        Plot individual clusters on each their own map.
        """
        cluster_counts = self.td.get_cluster_counts(var)
        n_clusters = np.min([len(self.td.get_clusters(var).cluster_ids), max_clusters])
        nrows = int(np.ceil(n_clusters / ncols))
        # fig, axs = south_pole_plots(nrows, ncols, h=nrows*3)
        # Intsead of south_pole_plots:

        projection = ccrs.SouthPolarStereo() if south_pole else ccrs.PlateCarree()
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, nrows*2.5), subplot_kw={'projection': projection})
        for ax in axs.flat:
            ax.coastlines(resolution="110m", linewidth=0.5)
            if(south_pole):
                ax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
        
        for i, id in enumerate(self.td.get_cluster_ids(var)[:n_clusters]):
            ax = axs.flat[i]
            self.plot_cluster_on_map(var, ax=ax, cluster_id=id, color=color)
            ax.set_title(f"id {id} with {cluster_counts[id]} members", fontsize=10)


<<<<<<< HEAD
<<<<<<< HEAD
    def plot_cluster_time_series(self, var, cluster_id, ax=None, max_trajectories=1_000, **plot_kwargs):
        """
        Plot the time series of a cluster.
        """
        cells = self.td.get_cluster_timeseries(var, cluster_id)
=======
    def plot_cluster_time_series(self, var, cluster_id, ax=None, max_trajectories=1_000, plot_shifts=False, **plot_kwargs):
        """
        Plot the time series of a cluster.
        """
        cell = self.td.get_cluster_cell_data(var, cluster_id)
        if(plot_shifts):
            cell = [ts.get_shifts() for ts in cell]
        else:
            cell = [ts[var] for ts in cell]
>>>>>>> c6fc662 (Docstring and type fixes)
=======
    def plot_cluster_time_series(self, var, cluster_id, ax=None, max_trajectories=1_000, **plot_kwargs):
        """
        Plot the time series of a cluster.
        """
        cells = self.td.get_cluster_timeseries(var, cluster_id)
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
        
        if ax is None:
            fig, ax = plt.subplots()


        # Limit the number of trajectories to plot
<<<<<<< HEAD
<<<<<<< HEAD
        max_trajectories = np.min([max_trajectories, len(cells)])

        # Shuffle the cell to get a random sample
        order = np.arange(len(cells))
=======
        max_trajectories = np.min([max_trajectories, len(cell)])

        # Shuffle the cell to get a random sample
        order = np.arange(len(cell))
>>>>>>> c6fc662 (Docstring and type fixes)
=======
        max_trajectories = np.min([max_trajectories, len(cells)])

        # Shuffle the cell to get a random sample
        order = np.arange(len(cells))
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
        np.random.shuffle(order) 
        order = order[:max_trajectories]

        for i in order:
<<<<<<< HEAD
<<<<<<< HEAD
            cells[i].plot(ax=ax, **plot_kwargs)
        
        if max_trajectories < len(cells):
            ax.set_title(f'Random sample of {max_trajectories} from total {len(cells)} cell for {var} in cluster {cluster_id}')
        else:                                                                              
            ax.set_title(f'{len(cells)} timeseries for {var} in cluster {cluster_id}')
=======
            cell[i].plot(ax=ax, **plot_kwargs)
=======
            cells[i].plot(ax=ax, **plot_kwargs)
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
        
        if max_trajectories < len(cells):
            ax.set_title(f'Random sample of {max_trajectories} from total {len(cells)} cell for {var} in cluster {cluster_id}')
        else:                                                                              
<<<<<<< HEAD
            ax.set_title(f'{len(cell)} timeseries for {var} in cluster {cluster_id}')
>>>>>>> c6fc662 (Docstring and type fixes)
=======
            ax.set_title(f'{len(cells)} timeseries for {var} in cluster {cluster_id}')
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
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

