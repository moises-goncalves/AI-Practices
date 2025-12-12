"""
Data processing and manipulation utilities.

This module provides functions for:
- Run-length encoding/decoding of segmentation masks
- Volume data extraction and processing
- Data augmentation helpers
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Tuple, Union

# Try to import cv2, it's optional for testing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


def rle_encode(img: np.ndarray) -> str:
    """
    Encode binary mask using run-length encoding (RLE).

    RLE is a simple compression technique that stores sequences of data as
    (start_position, length) pairs, which is very efficient for binary masks.

    Args:
        img (np.ndarray): Binary mask where 1 indicates mask, 0 indicates background.

    Returns:
        str: Run-length encoded string in format "start1 length1 start2 length2 ..."

    Example:
        >>> mask = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])
        >>> rle = rle_encode(mask)
        >>> print(rle)  # "3 2 5 2"

    Note:
        The encoding is 1-indexed (starts at position 1, not 0) to match
        Kaggle competition format.
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle: str, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decode run-length encoded mask.

    Args:
        mask_rle (str): Run-length encoded string in format "start1 length1 start2 length2 ..."
        shape (Tuple[int, int]): Output shape as (height, width).

    Returns:
        np.ndarray: Binary mask with specified shape where 1 indicates mask, 0 indicates background.

    Example:
        >>> rle_string = "3 2 5 2"
        >>> mask = rle_decode(rle_string, (2, 4))
        >>> print(mask)
        # [[0, 0, 1, 1],
        #  [1, 1, 0, 0]]

    Note:
        The decoding assumes 1-indexed positions to match Kaggle format.
    """
    s = mask_rle.split()
    starts = np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)

    # Convert from 1-indexed to 0-indexed
    starts -= 1
    ends = starts + lengths

    # Create flat mask and fill in runs
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape(shape)


def get_volume_data(
    data: pd.DataFrame,
    step: int = 96,
    stride: int = 1,
    stride_cutoff: int = 200
) -> List[pd.DataFrame]:
    """
    Extract volume sequences from CT study data.

    This function splits long CT studies into fixed-length sequences (volumes)
    for model training. It handles variable-length studies by:
    1. Downsampling studies longer than stride_cutoff
    2. Creating overlapping windows of length step
    3. Interpolating when needed to match exact step size

    Args:
        data (pd.DataFrame): DataFrame with study metadata, must have 'study' and 'z_pos' columns.
        step (int): Number of slices per volume. Default is 96.
        stride (int): Stride for downsampling long studies. Default is 1.
        stride_cutoff (int): Studies longer than this will be downsampled. Default is 200.

    Returns:
        List[pd.DataFrame]: List of DataFrames, each representing one volume of 'step' slices.

    Technical Details:
        - Studies are grouped by 'study' ID
        - Slices are sorted by z_pos (axial position)
        - Long studies (>stride_cutoff slices) are downsampled by stride
        - Volumes are created using sliding windows
        - The last incomplete window is handled separately if present
        - Interpolation ensures each volume has exactly 'step' slices

    Example:
        >>> study_data = pd.DataFrame({
        ...     'study': [1, 1, 1, 1, 1],
        ...     'z_pos': [0, 1, 2, 3, 4],
        ...     'instance': [0, 1, 2, 3, 4]
        ... })
        >>> volumes = get_volume_data(study_data, step=3, stride=1)
        >>> len(volumes)  # 2 volumes (0-2 and 2-4)
    """
    volumes = []

    for study_id, study_group in tqdm(data.groupby('study'), desc="Processing studies"):
        # Sort by axial position
        idxs = np.argsort(study_group.z_pos)
        study_group = study_group.iloc[idxs]
        study_group = study_group.reset_index(drop=True)
        study_group.instance = list(range(len(study_group)))

        # Downsample long studies
        if len(study_group) > stride_cutoff:
            study_group = study_group[::stride]

        # Check if we need to handle the last incomplete window
        take_last = not str(len(study_group) / step).endswith('.0')

        # Create volumes using sliding windows
        started = False
        for i in range(len(study_group) // step):
            rows = study_group[i * step:(i + 1) * step]

            # Interpolate if needed
            if len(rows) != step:
                indices = np.linspace(0, len(rows) - 1, step).astype(int)
                rows = pd.DataFrame([rows.iloc[idx] for idx in indices])

            volumes.append(rows)
            started = True

        # Handle studies shorter than step size
        if not started:
            rows = study_group
            if len(rows) < step:
                # Interpolate to reach step size
                indices = np.linspace(0, len(rows) - 1, step).astype(int)
                rows = pd.DataFrame([rows.iloc[idx] for idx in indices])
            volumes.append(rows)

        # Handle last incomplete window
        if take_last:
            rows = study_group[-step:]
            if len(rows) == step:
                volumes.append(rows)

    return volumes


def process_volume(volume: np.ndarray, target_size: int = 128) -> torch.Tensor:
    """
    Process a CT volume for model input.

    This function prepares a volume for inference by:
    1. Resizing all slices to target_size x target_size
    2. Splitting into fixed-length sequences (32 slices each)
    3. Handling the last incomplete sequence
    4. Converting to PyTorch tensor

    Args:
        volume (np.ndarray): Input volume with shape (num_slices, height, width).
        target_size (int): Target spatial dimension for resizing. Default is 128.

    Returns:
        torch.Tensor: Processed volume with shape (num_sequences, 32, target_size, target_size).

    Technical Details:
        - Each slice is resized using bilinear interpolation
        - Volume is split into non-overlapping windows of 32 slices
        - Last incomplete window is zero-padded if necessary
        - Output is a float tensor suitable for model input

    Example:
        >>> volume = np.random.rand(100, 512, 512)  # 100 slices, 512x512
        >>> processed = process_volume(volume, target_size=128)
        >>> print(processed.shape)  # (4, 32, 128, 128)
        # First 3 sequences have 32 slices each (96 total)
        # Last sequence has 4 real slices + 28 zero-padded slices
    """
    if not CV2_AVAILABLE:
        raise ImportError("cv2 (opencv-python) is required for process_volume. Install it with: pip install opencv-python")

    # Resize all slices
    volume = np.stack([
        cv2.resize(slice_img, (target_size, target_size))
        for slice_img in volume
    ])

    # Split into 32-slice sequences
    volumes = []
    sequence_length = 32
    cuts = [(x, x + sequence_length) for x in np.arange(0, volume.shape[0], sequence_length)[:-1]]

    if cuts:
        # Create sequences from complete windows
        for cut in cuts:
            volumes.append(volume[cut[0]:cut[1]])

        volumes = np.stack(volumes)
    else:
        # Handle case where volume is shorter than sequence_length
        volumes = np.zeros((1, sequence_length, target_size, target_size), dtype=np.uint8)
        volumes[0, :len(volume)] = volume

    # Handle last incomplete window
    if cuts:
        remaining_slices = volume[cuts[-1][1]:]
        if len(remaining_slices) > 0:
            last_volume = np.zeros((1, sequence_length, target_size, target_size), dtype=np.uint8)
            last_volume[0, :len(remaining_slices)] = remaining_slices
            volumes = np.concatenate([volumes, last_volume])

    # Convert to tensor
    volumes = torch.as_tensor(volumes).float()

    return volumes
