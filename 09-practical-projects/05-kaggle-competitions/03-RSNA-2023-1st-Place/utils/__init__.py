"""
Utility modules for RSNA 2023 Abdominal Trauma Detection Competition.

This package contains common utilities used across the project including:
- DICOM image processing functions
- Data loading and preprocessing utilities
- Distributed training helpers
- Common transformations and augmentations
"""

__version__ = "1.0.0"
__author__ = "RSNA 2023 Competition Team"

from .dicom_utils import (
    glob_sorted,
    get_standardized_pixel_array,
    get_windowed_image,
    get_rescaled_image,
    load_volume,
)

from .data_utils import (
    rle_encode,
    rle_decode,
    get_volume_data,
    process_volume,
)

from .distributed_utils import (
    is_dist_avail_and_initialized,
    get_world_size,
    get_rank,
    is_main_process,
    save_on_master,
    setup_for_distributed,
    init_distributed,
    seed_everything,
)

__all__ = [
    # DICOM utilities
    "glob_sorted",
    "get_standardized_pixel_array",
    "get_windowed_image",
    "get_rescaled_image",
    "load_volume",
    # Data utilities
    "rle_encode",
    "rle_decode",
    "get_volume_data",
    "process_volume",
    # Distributed utilities
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "save_on_master",
    "setup_for_distributed",
    "init_distributed",
    "seed_everything",
]
