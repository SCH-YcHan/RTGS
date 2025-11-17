# RTGS

Real Time Glomeruli Segmentation

This project provides a mmsegmentation-based U-Net pipeline for glomeruli segmentation in renal whole slide images (WSIs). Phase 1 implements batch WSI inference, and a design stub for Phase 2 (realtime viewer capture) is included for future expansion.

## Environment setup (Windows 11 + Conda example)

```bash
# create environment
conda create -n glom_wsi python=3.10
conda activate glom_wsi

# install dependencies
pip install -r requirements.txt

# IMPORTANT: install PyTorch/torchvision for your CUDA version separately
# follow https://pytorch.org/get-started/locally/
```

If you keep a local `mmsegmentation/` clone inside the project root, the scripts automatically add it to `PYTHONPATH`. Otherwise install mmsegmentation via `pip install -e mmsegmentation` or from PyPI.

## Batch WSI inference (Phase 1)

Run segmentation on a single WSI:

```bash
python -m wsi_tools.infer_wsi \
    --wsi_path ./data/sample_kidney.svs \
    --checkpoint ./checkpoints/glom_unet.pth \
    --config ./configs/glom_unet_wsi.py \
    --out_dir ./outputs_wsi \
    --tile_size 1024 \
    --tile_overlap 256 \
    --level 0 \
    --device cuda
```

### Arguments
- `--wsi_path`: input WSI path (.svs, .tif, .tiff).
- `--checkpoint`: trained mmsegmentation checkpoint.
- `--config`: mmsegmentation config file.
- `--out_dir`: directory for outputs (mask, overlay).
- `--tile_size`: square tile size in pixels (default 1024).
- `--tile_overlap`: overlap in pixels between adjacent tiles (default 256).
- `--level`: pyramid level to read (0 = full resolution).
- `--device`: `cuda` or `cpu`; defaults to CUDA when available.
- `--save_raw_mask`: also save a raw `.npy` mask array.
- `--downsample_for_overlay`: downsample factor for overlay export (default 4).

### Outputs
- `*_mask.png`: binary mask stitched for the requested WSI level (255 = glomerulus).
- `*_overlay.png`: RGB WSI image with a semi-transparent red overlay where glomeruli are predicted. Downsampled when `--downsample_for_overlay > 1` to reduce memory usage.

### Limitations & notes
- WSI formats supported by `openslide-python` are preferred (.svs, .tif). Other formats may require additional loaders.
- GPU acceleration is recommended; CPU-only inference can be slow on large slides.
- Adjust `tile_size` and `tile_overlap` to balance speed and boundary quality. Larger tiles reduce stitching seams but consume more memory.

## Future work (Phase 2)

`wsi_tools/realtime_stub.py` outlines how a realtime helper could capture a slide viewer window (e.g., via `mss`, `PIL.ImageGrab`, or Win32 APIs), run segmentation, and display overlays live. The stub documents the expected control flow and TODOs for implementing capture, batching, and rendering.

## Repository layout
- `wsi_tools/infer_wsi.py`: CLI for batch WSI segmentation.
- `wsi_tools/wsi_utils.py`: tiling, stitching, overlay, and WSI IO utilities.
- `wsi_tools/realtime_stub.py`: placeholder for realtime capture/overlay tooling.
- `configs/`: mmsegmentation configs (user-provided).
- `checkpoints/`: trained model weights (user-provided).
- `data/`: sample WSIs (optional; user-provided).
