"""HEALPix grid metadata for TOAD (ring scheme, power-of-two nside)."""

from __future__ import annotations

from dataclasses import dataclass

import healpix_geo.auto as hp_auto
import numpy as np


def nside_to_depth(nside: int) -> int:
    """Convert TOAD ring ``nside`` (power of 2) to healpix-geo ``depth``."""
    if nside <= 0 or not np.log2(nside).is_integer():
        raise ValueError(f"nside must be a positive power of 2, got {nside}")
    return int(np.log2(nside))


def depth_to_nside(depth: int) -> int:
    """Convert healpix-geo ``depth`` to TOAD ring ``nside``."""
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    return 1 << int(depth)


@dataclass(frozen=True)
class HealpixGrid:
    """Ring-order HEALPix grid with TOAD-facing ``nside``."""

    nside: int
    ellipsoid: str = "sphere"

    def __post_init__(self) -> None:
        nside_to_depth(self.nside)

    @property
    def depth(self) -> int:
        return nside_to_depth(self.nside)

    @property
    def npix(self) -> int:
        return 12 * self.nside**2

    def to_geo_grid(self) -> hp_auto.Grid:
        return hp_auto.Grid(
            level=self.depth,
            indexing_scheme="ring",
            ellipsoid=self.ellipsoid,
        )
