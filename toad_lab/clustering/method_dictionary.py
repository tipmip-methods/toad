from . import dbscan
from . import hdbscan

# Each new clustering detection procedure needs to register the function which
# maps the analysis to xr.DataArray 
clustering_methods = {
    'dbscan': dbscan.dbscan,
    'hdbscan': hdbscan.hdbscan
} 