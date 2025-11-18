"""Batch inference entrypoint for glomeruli segmentation on renal WSIs."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

import importlib.util
import numpy as np
from PIL import Image

from wsi_tools import wsi_utils

# Resolve project roots and make sure mmsegmentation is importable when it exists locally
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
MMSEG_ROOT = (REPO_ROOT / "mmsegmentation").resolve()
if MMSEG_ROOT.exists() and str(MMSEG_ROOT) not in sys.path:
    sys.path.insert(0, str(MMSEG_ROOT))

LOGGER = logging.getLogger(__name__)
DEFAULT_TILE_SIZE = 1024
DEFAULT_TILE_OVERLAP = 256
DEFAULT_OVERLAY_DOWNSAMPLE = 4
GLOM_CLASS_INDEX = 1


def _require_module(module_name: str) -> None:
    """Provide a friendly error if a required module is missing."""

    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Module '{module_name}' is required but not installed. Install it via pip/conda or adjust PYTHONPATH."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to input WSI (e.g., .svs, .tif).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to mmsegmentation checkpoint file.")
    parser.add_argument("--config", type=str, required=True, help="Path to mmsegmentation config file.")
    parser.add_argument("--out_dir", type=str, default="./outputs_wsi", help="Directory to store outputs.")
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE, help="Tile size for WSI tiling (pixels).")
    parser.add_argument("--tile_overlap", type=int, default=DEFAULT_TILE_OVERLAP, help="Overlap between tiles (pixels).")
    parser.add_argument("--level", type=int, default=0, help="WSI pyramid level to process (0 = highest resolution).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Computation device. Default selects CUDA if available, else CPU.",
    )
    parser.add_argument("--save_raw_mask", action="store_true", help="Also save the raw mask as a .npy file.")
    parser.add_argument(
        "--downsample_for_overlay",
        type=int,
        default=DEFAULT_OVERLAY_DOWNSAMPLE,
        help="Downsample factor when exporting overlay to reduce memory usage.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def select_device(user_device: str | None) -> str:
    _require_module("torch")
    import torch

    if user_device is not None:
        return user_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(config_path: Path, checkpoint_path: Path, device: str):
    """Load an mmsegmentation model with compatibility-friendly messaging."""

    _require_module("mmseg")
    _require_module("mmcv")
    # Some mmseg builds require ftfy at import time for tokenizer utilities.
    try:
        _require_module("ftfy")
    except ImportError as err:
        raise ImportError(
            "Module 'ftfy' is required by mmsegmentation but is missing. "
            "Install it with `pip install ftfy` and rerun."
        ) from err

    import mmcv

    LOGGER.info("Detected mmcv version: %s", getattr(mmcv, "__version__", "unknown"))
    try:
        from mmseg.apis import init_model
    except AssertionError as err:
        # mmseg raises AssertionError when the installed mmcv version is outside
        # its supported range. Surface a clearer guidance for users.
        raise RuntimeError(
            "mmsegmentation reported an incompatible mmcv version. "
            "Please reinstall mmcv to a version supported by your mmsegmentation package. "
            "The original error was: "
            f"{err}"
        ) from err
    except ModuleNotFoundError as err:
        if err.name == "ftfy":
            raise ImportError(
                "mmsegmentation could not import because 'ftfy' is missing. "
                "Install it with `pip install ftfy` and retry."
            ) from err
        raise

    model = init_model(str(config_path), str(checkpoint_path), device=device)
    return model


def infer_tile(model: Any, tile_img: np.ndarray) -> np.ndarray:
    """Run segmentation on a single tile and return a binary mask."""

    from mmseg.apis import inference_model

    result = inference_model(model, tile_img)
    if hasattr(result, "pred_sem_seg"):
        seg = result.pred_sem_seg.data.squeeze().cpu().numpy()
    elif isinstance(result, np.ndarray):
        seg = result
    else:
        raise TypeError(f"Unexpected inference output type: {type(result)}")

    if seg.ndim == 3:
        if seg.shape[0] == 1:
            seg = seg[0]
        elif seg.shape[2] == 1:
            seg = seg[:, :, 0]
        else:
            seg = np.argmax(seg, axis=0)
    mask = (seg == GLOM_CLASS_INDEX).astype(np.uint8)
    return mask


def main() -> None:
    args = parse_args()
    setup_logging()

    wsi_path = Path(args.wsi_path)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wsi_path.exists():
        raise FileNotFoundError(f"WSI file not found: {wsi_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)

    LOGGER.info("Loading model from %s with config %s", checkpoint_path, config_path)
    model = load_model(config_path, checkpoint_path, device=device)

    LOGGER.info("Opening WSI: %s", wsi_path)
    slide, meta = wsi_utils.open_wsi(str(wsi_path))
    LOGGER.info(
        "Slide levels: %d, level dimensions: %s", meta["level_count"], meta["level_dimensions"]
    )

    LOGGER.info(
        "Generating tiles at level %d (tile size %d, overlap %d)",
        args.level,
        args.tile_size,
        args.tile_overlap,
    )
    tiles = wsi_utils.generate_tiles(slide, args.level, args.tile_size, args.tile_overlap)
    LOGGER.info("Running inference on %d tiles...", len(tiles))

    _require_module("torch")
    import torch

    tile_masks: List[Dict[str, Any]] = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for tile in tiles:
            mask = infer_tile(model, tile["image"])
            tile_masks.append({"x": tile["x"], "y": tile["y"], "mask": mask})

    level_size = slide.level_dimensions[args.level]
    LOGGER.info("Stitching %d tile masks into full resolution mask...", len(tile_masks))
    full_mask = wsi_utils.stitch_tiles(tile_masks, level_size, args.tile_size, args.tile_overlap)

    mask_export = (full_mask * 255).astype(np.uint8)
    mask_path = out_dir / f"{wsi_path.stem}_mask.png"
    wsi_utils.save_large_image(str(mask_path), mask_export)

    if args.save_raw_mask:
        raw_path = out_dir / f"{wsi_path.stem}_mask.npy"
        np.save(raw_path, full_mask)
        LOGGER.info("Saved raw mask array to %s", raw_path)

    overlay_image = wsi_utils.get_level_image(slide, args.level, args.downsample_for_overlay)
    if args.downsample_for_overlay > 1:
        new_w, new_h = overlay_image.shape[1], overlay_image.shape[0]
        mask_for_overlay = np.array(
            Image.fromarray(full_mask).resize((new_w, new_h), resample=Image.Resampling.NEAREST)
        )
    else:
        mask_for_overlay = full_mask
    overlay = wsi_utils.overlay_mask_on_image(overlay_image, mask_for_overlay, color=(255, 0, 0), alpha=0.4)
    overlay_path = out_dir / f"{wsi_path.stem}_overlay.png"
    wsi_utils.save_large_image(str(overlay_path), overlay)

    LOGGER.info("WSI segmentation complete.")
    LOGGER.info("Mask saved at: %s", mask_path)
    LOGGER.info("Overlay saved at: %s", overlay_path)


if __name__ == "__main__":
    main()
