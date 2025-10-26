from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger

from karawm.configs import DEFAULT_KARAMARK_REMOVE_MODEL
from karawm.iopaint.const import DEFAULT_MODEL_DIR
from karawm.iopaint.download import scan_models
from karawm.iopaint.model_manager import ModelManager
from karawm.iopaint.schema import InpaintRequest

from karawm.karamark_detector import KaramarkDetector


class _SimpleTracker:
    def __init__(self, alpha: float = 0.75, iou_keep: float = 0.05):
        self.alpha = alpha
        self.iou_keep = iou_keep
        self.prev: tuple[int, int, int, int] | None = None

    @staticmethod
    def _iou(a, b) -> float:
        if a is None or b is None:
            return 0.0
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        xi0, yi0 = max(ax0, bx0), max(ay0, by0)
        xi1, yi1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, xi1 - xi0) * max(0, yi1 - yi0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def update(self, new_box: tuple[int, int, int, int] | None):
        if new_box is None:
            return self.prev
        if self.prev is None:
            self.prev = new_box
            return self.prev
        if self._iou(self.prev, new_box) < self.iou_keep:
            self.prev = new_box
            return self.prev
        px0, py0, px1, py1 = self.prev
        nx0, ny0, nx1, ny1 = new_box
        sx0 = int(self.alpha * nx0 + (1 - self.alpha) * px0)
        sy0 = int(self.alpha * ny0 + (1 - self.alpha) * py0)
        sx1 = int(self.alpha * nx1 + (1 - self.alpha) * px1)
        sy1 = int(self.alpha * ny1 + (1 - self.alpha) * py1)
        self.prev = (sx0, sy0, sx1, sy1)
        return self.prev


class _ShapeFallback:
    def __init__(self, prev_expand=2.8, area_main=(800, 3000), area_fb=(500, 6000)):
        self.prev_expand = prev_expand
        self.area_main = area_main
        self.area_fb = area_fb

    @staticmethod
    def _clip(x0, y0, x1, y1, w, h):
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        return x0, y0, x1, y1

    @staticmethod
    def _bbox_from_cnt(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, x + w, y + h)

    @staticmethod
    def _center(b):
        x0, y0, x1, y1 = b
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    def detect(self, img_bgr, prev_bbox=None):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        low, high = int(round(208 * 0.9)), int(round(208 * 1.1))
        mask = ((gray >= low) & (gray <= high)).astype(np.uint8) * 255

        bw = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, -5)
        bw = cv2.bitwise_and(bw, mask)

        r_tl = (0, 0, int(0.2 * w), int(0.2 * h))
        r_bl = (0, int(0.8 * h), int(0.2 * w), h)
        y0, y1 = int(0.40 * h), int(0.60 * h)

        region_mask = np.zeros_like(bw, dtype=np.uint8)
        for x0, ys, x1, ye in (r_tl, r_bl):
            region_mask[ys:ye, x0:x1] = 255
        region_mask[y0:y1, :] = 255

        if prev_bbox is not None:
            px0, py0, px1, py1 = prev_bbox
            pw, ph = (px1 - px0), (py1 - py0)
            cx, cy = self._center(prev_bbox)
            rw, rh = int(pw * self.prev_expand), int(ph * self.prev_expand)
            rx0, ry0 = cx - rw // 2, cy - rh // 2
            rx1, ry1 = cx + rw // 2, cy + rh // 2
            rx0, ry0, rx1, ry1 = self._clip(rx0, ry0, rx1, ry1, w, h)
            region_mask[ry0:ry1, rx0:rx1] = 255

        bw_r = cv2.bitwise_and(bw, region_mask)
        contours, _ = cv2.findContours(bw_r, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        cand = []
        for c in contours:
            a = cv2.contourArea(c)
            if self.area_main[0] <= a <= self.area_main[1]:
                peri = cv2.arcLength(c, True)
                if peri > 0:
                    circ = 4.0 * np.pi * a / (peri * peri)
                    if 0.55 <= circ <= 0.95:
                        cand.append(c)

        if not cand:
            return None

        if prev_bbox is None:
            best = max(cand, key=lambda c: cv2.contourArea(c))
        else:
            pcx, pcy = self._center(prev_bbox)
            best = max(cand,
                       key=lambda c: -(
                           ((self._bbox_from_cnt(c)[0] +
                             self._bbox_from_cnt(c)[2]) // 2 - pcx) ** 2
                           + ((self._bbox_from_cnt(c)[1] +
                               self._bbox_from_cnt(c)[3]) // 2 - pcy) ** 2
                       ))

        return self._bbox_from_cnt(best)


def _bbox_to_mask(h: int, w: int,
                  bbox: tuple[int, int, int, int] | None,
                  feather=14) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        return mask
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    mask[y0:y1, x0:x1] = 255
    k = feather if feather % 2 == 1 else feather + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


class KaramarkCleaner:
    def __init__(self, yolo_conf: float = 0.10,
                 feather: int = 14,
                 device: str | None = None):

        self.model_name = DEFAULT_KARAMARK_REMOVE_MODEL
        self.device = torch.device(device) if device else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scanned = scan_models()
        if self.model_name not in [it.name for it in scanned]:
            logger.info(f"{self.model_name} not found in {DEFAULT_MODEL_DIR}")

        self.model_manager = ModelManager(name=self.model_name,
                                          device=self.device)
        self.inpaint_req = InpaintRequest()

        self.detector = KaramarkDetector(conf_thres=yolo_conf)
        self.tracker = _SimpleTracker()
        self.fallback = _ShapeFallback()
        self.feather = feather

    def clean(self, input_image: np.ndarray,
              karamark_mask: np.ndarray) -> np.ndarray:
        result = self.model_manager(input_image, karamark_mask,
                                   self.inpaint_req)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def clean_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        H, W = frame_bgr.shape[:2]

        det = self.detector.detect(frame_bgr)
        yolo_box = det["bbox"] if det and det.get("detected", False) else None

        pick = yolo_box if yolo_box is not None else \
            self.fallback.detect(frame_bgr, self.tracker.prev)

        smooth_box = self.tracker.update(pick)

        if smooth_box is None:
            return frame_bgr

        mask = _bbox_to_mask(H, W, smooth_box, feather=self.feather)
        out = self.model_manager(frame_bgr, mask, self.inpaint_req)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
