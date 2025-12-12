"""
DICOM image processing utilities.

This module provides functions for loading, processing, and windowing DICOM medical images.
It includes optimized methods for batch processing of CT scan volumes.
"""

import numpy as np
from glob import glob
from typing import List, Union

# Try to import dicomsdl, it's optional for testing
try:
    import dicomsdl
    DICOMSDL_AVAILABLE = True
except ImportError:
    DICOMSDL_AVAILABLE = False
    dicomsdl = None


def _patch_dicomsdl():
    """
    Patches dicomsdl DataSet class to add numpy conversion method.

    This function extends the dicomsdl library by adding a convenient method
    to convert DICOM pixel data directly to numpy arrays. This is called
    automatically when the module is imported.
    """
    if not DICOMSDL_AVAILABLE:
        return

    def __dataset__to_numpy_image(self, index=0):
        """
        Convert DICOM pixel data to numpy array.

        Args:
            index (int): Frame index for multi-frame DICOM files. Default is 0.

        Returns:
            np.ndarray: Pixel array as numpy array.

        Raises:
            RuntimeError: If SamplesPerPixel != 1 (e.g., RGB images).
        """
        info = self.getPixelDataInfo()
        dtype = info['dtype']

        if info['SamplesPerPixel'] != 1:
            raise RuntimeError(
                'SamplesPerPixel != 1. This function only supports grayscale images.'
            )

        shape = [info['Rows'], info['Cols']]
        outarr = np.empty(shape, dtype=dtype)
        self.copyFrameData(index, outarr)
        return outarr

    dicomsdl._dicomsdl.DataSet.to_numpy_image = __dataset__to_numpy_image


# Apply patch when module is imported (only if dicomsdl is available)
_patch_dicomsdl()


def glob_sorted(path: str) -> List[str]:
    """
    Glob files and sort them by numeric filename.

    This function is specifically designed for DICOM files where filenames
    are typically numeric instance numbers.

    Args:
        path (str): Glob pattern to match files.

    Returns:
        List[str]: Sorted list of file paths.

    Example:
        >>> files = glob_sorted("/path/to/dicoms/*.dcm")
        >>> # Returns files sorted like: 1.dcm, 2.dcm, ..., 100.dcm
    """
    return sorted(
        glob(path),
        key=lambda x: int(x.split('/')[-1].split('.')[0])
    )


def get_standardized_pixel_array(dcm) -> np.ndarray:
    """
    Extract and standardize DICOM pixel array.

    Corrects DICOM pixel values if PixelRepresentation is 1 (signed),
    ensuring proper handling of signed integer data.

    Args:
        dcm: dicomsdl DataSet object.

    Returns:
        np.ndarray: Corrected pixel array.

    Technical Details:
        When PixelRepresentation == 1, the data uses signed integers.
        This function performs bit shifting to correctly interpret the values:
        1. Left shift by (BitsAllocated - BitsStored)
        2. Right shift by the same amount
        This ensures proper sign extension.
    """
    pixel_array = dcm.to_numpy_image()

    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        pixel_array = (pixel_array << bit_shift).astype(dtype) >> bit_shift

    return pixel_array


def get_rescaled_image(dcm) -> np.ndarray:
    """
    Apply DICOM rescale slope and intercept to convert to Hounsfield Units.

    This function converts stored pixel values to calibrated Hounsfield Units (HU)
    using the formula: HU = pixel_value * RescaleSlope + RescaleIntercept

    Args:
        dcm: dicomsdl DataSet object with RescaleSlope and RescaleIntercept attributes.

    Returns:
        np.ndarray: Image in Hounsfield Units.

    Note:
        Hounsfield Units are a standardized scale for CT imaging where:
        - Air: -1000 HU
        - Water: 0 HU
        - Bone: +1000 HU and above
    """
    rescale_intercept = dcm.RescaleIntercept
    rescale_slope = dcm.RescaleSlope

    img = dcm.to_numpy_image()
    img = rescale_slope * img + rescale_intercept

    return img


def get_windowed_image(
    img: np.ndarray,
    WL: int = 50,
    WW: int = 400
) -> np.ndarray:
    """
    Apply CT windowing to enhance visualization of specific tissue types.

    Windowing is a technique to adjust the contrast of CT images to highlight
    specific anatomical structures. This implementation uses soft-tissue windowing
    by default (WL=50, WW=400).

    Args:
        img (np.ndarray): Input image in Hounsfield Units.
        WL (int): Window Level (center of the window). Default is 50.
        WW (int): Window Width (range of HU values to display). Default is 400.

    Returns:
        np.ndarray: Windowed image normalized to 0-255 range as uint8.

    Technical Details:
        Window Level (WL): The center HU value of the viewing window
        Window Width (WW): The range of HU values to display
        Upper bound = WL + WW/2
        Lower bound = WL - WW/2

    Common Window Settings:
        - Soft tissue: WL=40-60, WW=350-400
        - Lung: WL=-500, WW=1500
        - Bone: WL=300, WW=1500
        - Brain: WL=40, WW=80
    """
    upper = WL + WW // 2
    lower = WL - WW // 2

    # Clip values to window range
    X = np.clip(img.copy(), lower, upper)

    # Normalize to 0-1 range
    X = X - np.min(X)
    X = X / np.max(X)

    # Scale to 0-255 and convert to uint8
    X = (X * 255.0).astype('uint8')

    return X


def load_volume(
    dcm_paths: List[str],
    apply_windowing: bool = True,
    WL: int = 50,
    WW: int = 400
) -> np.ndarray:
    """
    Load a complete CT volume from DICOM files.

    This function loads multiple DICOM slices and stacks them into a 3D volume.
    Each slice is rescaled to Hounsfield Units and optionally windowed.

    Args:
        dcm_paths (List[str]): List of paths to DICOM files for each slice.
        apply_windowing (bool): Whether to apply CT windowing. Default is True.
        WL (int): Window Level if windowing is applied. Default is 50.
        WW (int): Window Width if windowing is applied. Default is 400.

    Returns:
        np.ndarray: 3D volume with shape (num_slices, height, width).
                   Values are normalized to 0-1 range if windowing is applied.

    Example:
        >>> dcm_files = glob_sorted("/path/to/study/*.dcm")
        >>> volume = load_volume(dcm_files, apply_windowing=True)
        >>> print(volume.shape)  # (num_slices, 512, 512)

    Note:
        This function handles negative pixel values by adding the absolute
        minimum value before normalization.
    """
    volume = []

    for dcm_path in dcm_paths:
        dcm = dicomsdl.open(dcm_path)

        # Get image in Hounsfield Units
        image = get_rescaled_image(dcm)

        # Apply windowing if requested
        if apply_windowing:
            image = get_windowed_image(image, WL, WW)

        # Handle negative values
        if np.min(image) < 0:
            image = image + np.abs(np.min(image))

        # Normalize to 0-1 range
        image = image / image.max()

        volume.append(image)

    return np.stack(volume)
