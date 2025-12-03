"""Warning configuration for TOAD package.

This module configures warning filters that must be applied before importing
dependencies that may trigger warnings (e.g., cartopy/pyproj).
"""

import warnings

# Suppress pyproj network warning (harmless - only affects PROJ database updates)
# This warning occurs when pyproj cannot set up network access for PROJ database
# updates. It doesn't affect functionality, only prevents automatic updates.
warnings.filterwarnings(
    "ignore",
    message=".*pyproj unable to set PROJ database path.*",
    category=UserWarning,
)
