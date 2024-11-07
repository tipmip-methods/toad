from . import asdetect

# Each new abrupt shift detection method needs to register the function which
# maps the analysis to xr.DataArray 
detection_methods = {
    'asdetect': asdetect.detect
} 