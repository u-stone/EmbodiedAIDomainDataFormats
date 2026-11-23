"""Concrete low-level readers and inspectors for the 5 mainstream Embodied AI dataset formats."""

from .hdf5_format import HDF5FormatReader
from .lerobot_v3_format import LeRobotV3FormatReader
from .mcap_format import MCAPFormatReader
from .rlds_format import RLDSFormatReader
from .zarr_format import ZarrFormatReader

FORMAT_REGISTRY = {
    "mcap": MCAPFormatReader,
    "hdf5": HDF5FormatReader,
    "zarr": ZarrFormatReader,
    "rlds": RLDSFormatReader,
    "lerobot_v3": LeRobotV3FormatReader,
}

__all__ = [
    "FORMAT_REGISTRY",
    "MCAPFormatReader",
    "HDF5FormatReader",
    "ZarrFormatReader",
    "RLDSFormatReader",
    "LeRobotV3FormatReader",
]
