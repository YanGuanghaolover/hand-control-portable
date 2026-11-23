# -*- coding: utf-8 -*-
"""Shared HUD text rendering helpers.

Keeping these helpers independent from the application entry points prevents
``draw.py`` and ``calibrate.py`` from importing the whole controller just to
render Chinese text.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np


class TextCache:
    """Pre-render and cache text patches for the OpenCV HUD."""

    FONTS = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]

    def __init__(self, size: int = 18, max_entries: int = 256) -> None:
        from PIL import ImageFont

        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.font = None
        for path in self.FONTS:
            if os.path.exists(path):
                try:
                    self.font = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
        self.cache: Dict[tuple, np.ndarray] = {}
        self.max_entries = max_entries
        self._pad = 4

    @property
    def ok(self) -> bool:
        return self.font is not None

    def get(self, text: str, color=(255, 255, 255), bg=None) -> Optional[np.ndarray]:
        if not self.ok:
            return None
        key = (text, color, bg)
        if key in self.cache:
            return self.cache[key]

        from PIL import Image, ImageDraw

        probe = Image.new("RGBA", (8, 8))
        bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=self.font)
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        background = (0, 0, 0, 0) if bg is None else (bg[2], bg[1], bg[0], 255)
        image = Image.new(
            "RGBA", (width + self._pad * 2, height + self._pad * 2), background
        )
        ImageDraw.Draw(image).text(
            (self._pad - bbox[0], self._pad - bbox[1]),
            text,
            font=self.font,
            fill=(color[2], color[1], color[0], 255),
        )
        patch = np.array(image)
        if len(self.cache) >= self.max_entries:
            # HUD strings are normally static. Bound the cache so a caller that
            # includes changing values cannot grow it for the whole session.
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = patch
        return patch


def blend(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend an RGBA patch onto a BGR frame, clipping at the edges."""
    patch_h, patch_w = patch.shape[:2]
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + patch_w), min(frame_h, y + patch_h)
    if x0 >= x1 or y0 >= y1:
        return
    sub = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (
        sub[:, :, :3].astype(np.float32) * alpha + roi * (1 - alpha)
    ).astype(np.uint8)


def put_cn(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    text_cache: TextCache,
    color=(255, 255, 255),
    bg=None,
) -> None:
    """Draw Chinese text, falling back to ASCII when no font is available."""
    patch = text_cache.get(text, color, bg)
    if patch is not None:
        blend(frame, patch, x, y)
    else:
        ascii_text = text.encode("ascii", "ignore").decode()
        cv2.putText(
            frame,
            ascii_text,
            (x, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
