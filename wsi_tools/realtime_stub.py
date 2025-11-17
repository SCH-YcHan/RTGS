"""Skeleton for future realtime glomeruli overlay helper."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def capture_viewer_region() -> np.ndarray:
    """Capture the currently visible slide viewer region.

    TODO: Implement platform-specific screen capture. Potential options:
    - ``mss`` for fast multi-platform screen grabs.
    - ``PIL.ImageGrab`` for a simple, dependency-light approach on Windows.
    - ``pywin32`` or ``win32gui`` for capturing a specific window handle when
      using Windows-native slide viewers.
    """

    raise NotImplementedError("Screen capture is not implemented yet. See TODO notes in the docstring.")


def infer_single_frame(model: Any, frame: np.ndarray, device: str) -> np.ndarray:
    """Run segmentation on a single captured frame.

    Planned approach:
    - Option A: Resize the frame to the model's expected input size and run a
      single forward pass. Fastest, but lower detail.
    - Option B: Reuse the tiling utilities from ``wsi_utils`` to preserve
      resolution at the cost of throughput. For realtime usage this likely
      needs batching and asynchronous inference.
    """

    raise NotImplementedError("Realtime frame inference is a future addition.")


def run_realtime_loop(model: Any, device: str) -> None:
    """Continuously capture the viewer and display overlay results.

    Expected control flow (high level):
    1. Configure capture region/window handle.
    2. While loop capturing frames at a target FPS (e.g., 1-5 fps initially).
    3. Perform ``infer_single_frame`` on each frame.
    4. Visualize the overlay in a separate window using ``cv2.imshow`` or a
       lightweight GUI toolkit. Consider throttling display updates to avoid UI
       lag.
    5. Provide a keyboard interrupt/break condition for clean shutdown.

    Performance considerations and TODOs:
    - Introduce async/background inference to avoid blocking capture.
    - Allow configurable downsampling for smoother previews.
    - Integrate with ``wsi_tools.wsi_utils`` when available to share tiling logic.
    """

    LOGGER.info("Realtime loop placeholder running on device=%s", device)
    raise NotImplementedError("Realtime loop is a stub. Implement capture and rendering logic later.")
