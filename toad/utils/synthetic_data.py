import numpy as np
import xarray as xr
from scipy.stats import multivariate_normal


def create_global_dataset(
    lat_size=90,
    lon_size=180,
    time_size=80,
    background_trend=0.0,
    background_noise=0.005,
    n_shifts=3,
    random_seed=42,
):
    """
    Generate a global dataset with background trend and spatially coherent abrupt shifts.

    Args:
        lat_size (int): Number of latitude points
        lon_size (int): Number of longitude points
        time_size (int): Number of time points
        background_trend (float): Linear trend coefficient
        background_noise (float): Amplitude of random noise
        n_shifts (int): Number of abrupt shift events to add
        random_seed (int): Seed for reproducible results

    Returns:
        data (xr.Dataset): Dataset containing the time series and coordinates
        labels (xr.DataArray): Binary indicator of shift locations
        shift_params (dict): Parameters of the generated shifts

    Example:
        >>> data_ds, labels_xr, shift_params = create_global_dataset(
        >>>    lat_size=30,
        >>>    lon_size=60,
        >>>    time_size=120,
        >>>    n_shifts=3,
        >>>    random_seed=1,
        >>>    background_noise=0.01,
        >>> )
    """
    np.random.seed(random_seed)

    # Create coordinate arrays
    time = np.linspace(0, 100, time_size)
    lats = np.linspace(-90, 90, lat_size)
    lons = np.linspace(-180, 180, lon_size)

    # Generate background data with trend and noise
    data = np.zeros((time_size, lat_size, lon_size))
    for i in range(lat_size):
        for j in range(lon_size):
            data[:, i, j] = (
                background_trend * time + background_noise * np.random.randn(time_size)
            )

    # Create xarray DataArray
    data_xr = xr.DataArray(
        data,
        coords={"time": time, "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
    )

    # Generate distinct shift times
    shift_times = np.linspace(20, 70, n_shifts).astype(int)

    # Generate shift parameters
    center_lat = np.random.uniform(-80, 80, size=n_shifts)
    center_lon = np.random.uniform(-180, 180, size=n_shifts)
    sigma_lat = np.random.uniform(10, 20, size=n_shifts)
    sigma_lon = np.random.uniform(15, 40, size=n_shifts)
    sigma_time = np.ones(n_shifts)  # Sharp shifts
    steepness = np.random.uniform(1.5, 3.0, size=n_shifts)
    shift_magnitude = np.random.uniform(0.95, 1.05, size=n_shifts)

    shift_params = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "center_time": shift_times,
        "sigma_lat": sigma_lat,
        "sigma_lon": sigma_lon,
        "sigma_time": sigma_time,
        "steepness": steepness,
        "magnitude": shift_magnitude,
    }

    # Initialize labels array
    labels = np.zeros_like(data, dtype=int)

    # Add shifts to the data
    for i in range(n_shifts):
        data_xr, shift_labels = add_shift_blob(
            data_xr,
            center_lat=center_lat[i],
            center_lon=center_lon[i],
            center_time=shift_times[i],
            sigma_lat=sigma_lat[i],
            sigma_lon=sigma_lon[i],
            sigma_time=sigma_time[i],
            steepness=steepness[i],
            magnitude=shift_magnitude[i],
        )
        labels = np.logical_or(labels, shift_labels.values).astype(int)

    # Convert to dataset and add Cartesian coordinates
    data_ds = data_xr.to_dataset(name="value")
    data_ds = add_xyz_coords(data_ds)

    # Create label DataArray
    labels_xr = xr.DataArray(
        labels, coords=data_xr.coords, dims=data_xr.dims, name="shift_label"
    )

    return data_ds, labels_xr, shift_params


def add_shift_blob(
    data_xr,
    center_lat: float = 0,
    center_lon: float = 0,
    center_time: int = 50,
    sigma_lat: float = 10,
    sigma_lon: float = 20,
    sigma_time: float = 1,
    steepness: float = 2.0,
    magnitude: float = 1.0,
):
    """Add an abrupt shift blob to the data array."""
    lats = data_xr.coords["lat"].values
    lons = data_xr.coords["lon"].values
    time = data_xr.coords["time"].values

    # Handle longitude periodicity
    extended_lons = np.concatenate([lons - 360, lons, lons + 360])

    # Create grids
    lat_grid, extended_lon_grid = np.meshgrid(lats, extended_lons, indexing="ij")
    pos_extended = np.dstack((lat_grid, extended_lon_grid))

    # Calculate Gaussian blob
    cov = np.array([[sigma_lat**2, 0], [0, sigma_lon**2]])
    rv = multivariate_normal(mean=[center_lat, center_lon], cov=cov)
    extended_blob = rv.pdf(pos_extended)
    blob_normalized = extended_blob / np.max(extended_blob)

    # Create label array
    labels = np.zeros(data_xr.shape, dtype=int)

    def sigmoid(t, t0, k: float = 1.0) -> np.ndarray:
        """Sigmoid transition function"""
        return 1 / (1 + np.exp(-k * (t - t0)))

    # Apply shift to data
    for i in range(len(lats)):
        for j in range(len(lons)):
            # Sum contributions from periodic longitudes
            blob_contribution = np.sum(
                [blob_normalized[i, j + k * len(lons)] for k in range(3)]
            )

            if blob_contribution > 0.5:
                t_shift = int(center_time + np.random.normal(0, sigma_time))
                t_shift = max(0, min(t_shift, len(time) - 1))

                labels[t_shift, i, j] = 1

                shift_effect = (
                    magnitude
                    # * blob_contribution
                    * sigmoid(time, time[t_shift], steepness)
                )
                data_xr.values[:, i, j] += shift_effect

    labels_xr = xr.DataArray(labels, coords=data_xr.coords, dims=data_xr.dims)
    return data_xr, labels_xr


def add_xyz_coords(data):
    """Add 3D Cartesian coordinates to the dataset for globe plotting."""
    lat = data.coords["lat"].values
    lon = data.coords["lon"].values

    # WGS84 parameters
    a = 6378.137  # semi-major axis (km)
    b = 6356.752  # semi-minor axis (km)
    e2 = 1 - (b**2 / a**2)

    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing="ij")

    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)

    N = a / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)

    x = N * np.cos(lat_rad) * np.cos(lon_rad)
    y = N * np.cos(lat_rad) * np.sin(lon_rad)
    z = (b**2 / a**2 * N) * np.sin(lat_rad)

    x = x.T
    y = y.T
    z = z.T

    data = data.assign_coords(x=(("lat", "lon"), x))
    data = data.assign_coords(y=(("lat", "lon"), y))
    data = data.assign_coords(z=(("lat", "lon"), z))

    return data
