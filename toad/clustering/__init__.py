import logging
import xarray as xr
import numpy as np
from typing import Callable
from toad._version import __version__
import inspect
from typing import Optional, Union

from toad.clustering.prepare_data import prepare_dataframe
from sklearn.base import ClusterMixin


logger = logging.getLogger("TOAD")

def compute_clusters(
        data: xr.Dataset,
        var : str,
        method : ClusterMixin,
        shifts_filter_func: Callable[[float], bool],
        var_filter_func: Optional[Callable[[float], bool]] = None,
        shifts_label: Optional[str] = None,
<<<<<<< HEAD
        scaler: Optional[str] = 'StandardScaler',
=======
        scaler: str = 'StandardScaler',
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)
        output_label_suffix: str = "",
        overwrite: bool = False,
        merge_input: bool = True,
<<<<<<< HEAD
<<<<<<< HEAD
        sort_by_size: bool = True,
    ) -> Union[xr.Dataset, xr.DataArray]:
    """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm. 

        >> Args:
            var:
                Name of the variable in the dataset to cluster.
            method:
                The clustering method to use. Choose methods from `sklearn.cluster` or create your by inheriting from `sklearn.base.ClusterMixin`.
            shifts_filter_func:
                A callable used to filter the shifts before clustering, such as `lambda x: np.abs(x)>0.8`. 
            var_filter_func:
                A callable used to filter the primary variable before clustering. Defaults to None.
            shifts_label:
                Name of the variable containing precomputed shifts. Defaults to {var}_dts.
            scaler:
                The scaling method to apply to the data before clustering. Choose between 'StandardScaler', 'MinMaxScaler' and None. Defaults to 'StandardScaler'.
            output_label_suffix:
                A suffix to add to the output label. Defaults to "".
            overwrite:
                Whether to overwrite existing variable. Defaults to False.
            merge_input:
                Whether to merge the clustering results with the input dataset. Defaults to True.
            sort_by_size:
                Whether to reorder clusters by size. Defaults to True.

        >> Returns:
            xr.Dataset:
                If `merge_input` is `True`, returns an `xarray.Dataset` containing the original data and the clustering results.
            xr.DataArray:
                If `merge_input` is `False`, returns an `xarray.DataArray` containing the clustering results.

        >> Raises:
            ValueError:
                If data is invalid or required parameters are missing

        """
=======
=======
        sort_by_size: bool = True,
>>>>>>> 1787555 (Rename cluster ids by size after computation)
    ) -> Union[xr.Dataset, xr.DataArray]:
    """Apply clustering to a dataset's temporal shifts using a sklearn-compatible clustering algorithm. 

        Args:
            var: Name of the variable in the dataset to cluster.
            method: The clustering method to use. Choose methods from `sklearn.cluster` or create your by inheriting from `sklearn.base.ClusterMixin`.
            shifts_filter_func: A callable used to filter the shifts before clustering, such as `lambda x: np.abs(x)>0.8`. 
            var_filter_func: A callable used to filter the primary variable before clustering. Defaults to None.
            shifts_label: Name of the variable containing precomputed shifts. Defaults to {var}_dts.
            scaler: The scaling method to apply to the data before clustering. Choose between 'StandardScaler', 'MinMaxScaler' and None. Defaults to 'StandardScaler'.
            output_label_suffix: A suffix to add to the output label. Defaults to "".
            overwrite: Whether to overwrite existing variable. Defaults to False.
            merge_input: Whether to merge the clustering results with the input dataset. Defaults to True.
            sort_by_size: Whether to reorder clusters by size. Defaults to True.

        Returns:
            xr.Dataset: If `merge_input` is `True`, returns an `xarray.Dataset` containing the original data and the clustering results.
            xr.DataArray: If `merge_input` is `False`, returns an `xarray.DataArray` containing the clustering results.

        Raises:
            ValueError: If data is invalid or required parameters are missing

    """
>>>>>>> c6fc662 (Docstring and type fixes)
    # TODO: (1) Fix: should also return auxillary coordinates. For now only returns coords in dims. 

    # Check shifts var
    all_vars = list(data.data_vars.keys())
    shifts_label = shifts_label if shifts_label else f'{var}_dts'  # default to {var}_dts
    
    if shifts_label not in all_vars:
        raise ValueError(f'Shifts not found at {shifts_label}. Please run shifts on {var} first, or provide a custom "shifts_label"')
    if data[shifts_label].ndim != 3:
        raise ValueError('data must be 3-dimensional')
    
    # 1. Set output label
    output_label = f'{var}_cluster{output_label_suffix}'

    # Check if the output_label is already in the data
    if output_label in data and merge_input:
        if overwrite:
            logger.warning(f'Overwriting variable {output_label}')
            data = data.drop_vars(output_label)
        else:
            logger.warning(f'{output_label} already exists. Please pass overwrite=True to overwrite it.')
            return data

    # 2. Preprocessing

    # Prepare the data for clustering
    # filtered_data is a pandas df that contains the indeces of the data that passed the filters (var_func and dts_func)
    filtered_data, dims, importance_weights, scaled_coords = prepare_dataframe(
        data, var, shifts_label, var_filter_func, shifts_filter_func, scaler
    )

    # Save method params before clustering (because they might change during clustering)
    method_params = {
        f'method_{param}': str(value) 
        for param, value in dict(sorted(vars(method).items())).items() 
        if value is not None
    }

    # 3. Perform clustering
    logger.info(f'Applying clustering method {method}')
<<<<<<< HEAD
<<<<<<< HEAD
    cluster_labels = method.fit_predict(scaled_coords, importance_weights)
    cluster_labels = np.array(cluster_labels) # make sure it's a numpy array
<<<<<<< HEAD
=======
    clusters = method.fit_predict(scaled_coords, importance_weights)
>>>>>>> c6fc662 (Docstring and type fixes)
=======
    cluster_labels = method.fit_predict(scaled_coords, importance_weights)
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)
=======
>>>>>>> bc5ef07 (Fix: cluster_labels should be np.array)

    # 5. Convert back to xarray DataArray
    df_dims = data[dims].to_dataframe().reset_index()       # create a pandas df with original dims
    df_dims[output_label] = -1                              # Initialize cluster column with -1
    df_dims.loc[filtered_data.index, output_label] = cluster_labels # Assign cluster labels to the dataframe
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 1787555 (Rename cluster ids by size after computation)

    if sort_by_size:
        # Rename clusters by size (largest cluster -> 0, second largest -> 1, etc., keeping -1 for noise)
        valid_labels = cluster_labels[cluster_labels != -1]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        label_mapping = dict(zip(unique_labels[np.argsort(counts)[::-1]], range(len(unique_labels))))
        cluster_labels = np.array([label_mapping.get(label, -1) for label in cluster_labels])
        df_dims.loc[filtered_data.index, output_label] = cluster_labels
    
    # Convert to xarray
<<<<<<< HEAD
=======
>>>>>>> 341e8af ([Minor breaking changes] Enhancements to Cluster and Shifts Variable Handling)
=======
>>>>>>> 1787555 (Rename cluster ids by size after computation)
    clusters = df_dims.set_index(dims).to_xarray()          # Convert dataframe to xarray (DataSet)
    clusters = clusters[output_label]                       # select only cluster labels
    
    # 6. Save details as attributes
    clusters.attrs.update({
        f'cluster_ids': np.unique(cluster_labels).astype(int),
        f"var_filter_func": inspect.getsource(var_filter_func) if var_filter_func else "None",
        f"shifts_filter_func": inspect.getsource(shifts_filter_func) if shifts_filter_func else "None",
        f"scaler": scaler,
        f'method_name': method.__class__.__name__,
    })
    
    # Add saved params as attributes
    clusters.attrs.update(method_params)

    # add git version
    clusters.attrs['toad_version'] = __version__

    # 7. Merge the cluster labels back into the original data
    if merge_input:
        return xr.merge([data, clusters], combine_attrs="override") # xr.dataset
    else:
        return clusters # xr.dataarray


