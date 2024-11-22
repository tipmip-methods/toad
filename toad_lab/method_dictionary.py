from toad_lab.clustering.dbscan import dbscan
from toad_lab.clustering.hdbscan import hdbscan

# Each new clustering detection procedure needs to register the function which
# maps the analysis to xr.DataArray 
clustering_methods = {
    'dbscan': dbscan,
    'hdbscan': hdbscan
} 


from toad_lab.shifts_detection.asdetect import asdetect

# Each new abrupt shift detection method needs to register the function which
# maps the analysis to xr.DataArray 
shifts_methods = {
    'asdetect': asdetect
} 

