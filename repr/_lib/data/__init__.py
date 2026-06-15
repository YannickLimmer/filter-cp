"""Dataset loaders for every primary cell of the paper's main table.

Each loader returns a ``(Y, A)`` tuple and downloads from a public
source on first use into ``CAPFACTOR_REPR_CACHE``
(default ``~/.cache/capfactor_repr/``).  No credentials, no
proprietary data.
"""

from ._common import LoadedCell
from .aqi import load_aqi
from .elec import load_elec
from .ett import load_ett
from .jena import load_jena
from .largest import load_largest
from .loop_seattle import load_loop_seattle
from .metrla import load_metrla
from .pems_bay import load_pems_bay
from .solar import load_solar
from .synthetic import synthesise_gpvar


CELL_LOADERS = {
    "metrla": ("metrla", load_metrla, 20),
    "metrla_full": ("metrla", load_metrla, 207),
    "pems_bay": ("pems_bay", load_pems_bay, 50),
    "pems_bay_full": ("pems_bay", load_pems_bay, 325),
    "aqi": ("aqi", load_aqi, 12),
    "elec": ("elec", load_elec, 20),
    "ett": ("ett", load_ett, 7),
    "solar": ("solar", load_solar, 20),
    "jena": ("jena", load_jena, 18),
    "loop_seattle": ("loop_seattle", load_loop_seattle, 20),
    "largest": ("largest", load_largest, 2327),
    "synthetic": ("synthetic", synthesise_gpvar, 20),
}

__all__ = [
    "CELL_LOADERS", "LoadedCell",
    "load_metrla", "load_pems_bay", "load_aqi", "load_elec", "load_ett",
    "load_solar", "load_jena", "load_loop_seattle", "load_largest",
    "synthesise_gpvar",
]
