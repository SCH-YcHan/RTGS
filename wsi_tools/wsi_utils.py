"""Utility functions for working with renal whole slide images (WSIs).

These helpers are intentionally lightweight so they can be re-used for both
batch inference and future realtime capture flows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import importlib.util
import logging

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _require_module(module_name: str) -> None:
    """Ensure a module is importable before attempting to use it.

    The project discourages wrapping imports in try/except blocks. This helper
    provides a friendly error message before the import happens.
    """

    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Required module '{module_name}' is missing. Install it via pip or conda before running WSI tools."
        )


def open_wsi(path: str):
    """Open a WSI file and return the OpenSlide object and metadata.

    Parameters
    ----------
    path: str
        Path to the WSI file (.svs, .tif, .tiff, etc.).

    Returns
    -------
    slide: openslide.OpenSlide
        Opened slide object.
    meta: Dict[str, object]
        Metadata including dimensions and level information.
    """

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"WSI file not found: {path_obj}")

    _require_module("openslide")
    from openslide import OpenSlide  # imported here to keep module load light

    slide = OpenSlide(str(path_obj))
    level_dims = slide.level_dimensions
    meta: Dict[str, object] = {
        "filename": path_obj.name,
        "level_count": slide.level_count,
        "level_dimensions": level_dims,
        "level_downsamples": slide.level_downsamples,
        "base_dimensions": level_dims[0],
    }
    return slide, meta


def get_level_image(
    slide, level: int, downsample_factor_for_overlay: int = 1
) -> np.ndarray:
    """Return the RGB image at the specified pyramid level.

    The returned image can be optionally downsampled to save memory for
    visualization or overlay creation.
    """

    if level < 0 or level >= slide.level_count:
        raise ValueError(f"Requested level {level} is outside the slide's available range (0-{slide.level_count - 1}).")

    dims = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, dims).convert("RGB")
    image = np.array(region)
    if downsample_factor_for_overlay > 1:
        new_w = max(1, dims[0] // downsample_factor_for_overlay)
        new_h = max(1, dims[1] // downsample_factor_for_overlay)
        image = np.array(region.resize((new_w, new_h), resample=Image.Resampling.LANCZOS))
    return image


def generate_tiles(
    slide, level: int, tile_size: int, tile_overlap: int
) -> List[Dict[str, object]]:
    """Generate tiles for a given WSI level.

    Returns a list of dictionaries containing tile coordinates and the tile
    image as an RGB numpy array.
    """

    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if tile_overlap < 0:
        raise ValueError("tile_overlap must be non-negative")

    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile_size to ensure positive stride.")

    width, height = slide.level_dimensions[level]
    tiles: List[Dict[str, object]] = []
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            tile_w = min(tile_size, width - x)
            tile_h = min(tile_size, height - y)
            region = slide.read_region((x, y), level, (tile_w, tile_h)).convert("RGB")
            tile_img = np.array(region)
            tiles.append({
                "x": x,
                "y": y,
                "w": tile_w,
                "h": tile_h,
                "image": tile_img,
            })
    LOGGER.info("Generated %d tiles (level %d, size %d, overlap %d).", len(tiles), level, tile_size, tile_overlap)
    return tiles


def stitch_tiles(
    tile_masks: Iterable[Dict[str, object]],
    level_size: Sequence[int],
    tile_size: int,
    tile_overlap: int,
) -> np.ndarray:
    """Stitch tile-level masks into a full-resolution mask for the level."""

    width, height = level_size
    full_mask = np.zeros((height, width), dtype=np.uint8)

    for tile in tile_masks:
        x = int(tile["x"])
        y = int(tile["y"])
        mask = np.array(tile["mask"], dtype=np.uint8)
        tile_h, tile_w = mask.shape[:2]

        x_end = min(width, x + tile_w)
        y_end = min(height, y + tile_h)
        mask_crop = mask[: y_end - y, : x_end - x]

        current_region = full_mask[y:y_end, x:x_end]
        combined = np.maximum(current_region, mask_crop)
        full_mask[y:y_end, x:x_end] = combined

    return full_mask


def overlay_mask_on_image(
    image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.4
) -> np.ndarray:
    """Overlay a binary mask onto an RGB image using alpha blending."""

    if image.dtype != np.uint8:
        raise ValueError("image must be a uint8 RGB array")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    overlay = image.copy()
    mask_bool = mask > 0
    color_arr = np.zeros_like(image)
    color_arr[:, :, 0] = color[0]
    color_arr[:, :, 1] = color[1]
    color_arr[:, :, 2] = color[2]

    overlay[mask_bool] = (
        alpha * color_arr[mask_bool] + (1 - alpha) * overlay[mask_bool]
    ).astype(np.uint8)
    return overlay


def save_large_image(path: str, image: np.ndarray) -> None:
    """Save a potentially large numpy array image to disk using PIL."""

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(str(path_obj))
    LOGGER.info("Saved image to %s", path_obj)
