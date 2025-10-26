from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from loguru import logger

# --- IOPaint / LaMa (기존 의존) ---
from karawm.configs import DEFAULT_WATERMARK_REMOVE_MODEL
from karawm.iopaint.const import DEFAULT_MODEL_DIR
from karawm.iopaint.download import scan_models  # cli_download_model는 사용 안 함(수동 가중치 정책)
from karawm.iopaint.model_manager import ModelManager
from karawm.iopaint.schema import InpaintRequest

# --- YOLO Detector (기존 모듈 재사용) ---
#   클래스명/경로는 리포에 이미 존재하는 탐지 래퍼와 호환 (SoraWaterMarkDetector)
from karawm.watermark_detector import SoraWaterMarkDetector  # noqa: E402

# =========================================================
#                       TRACKING
# =========================================================
class _SimpleTracker:
    """IoU 체크 + EMA 스무딩"""
    def __init__(self, alpha: float = 0.75, iou_keep: float = 0.05):
        self.alpha = alpha
        self.iou_keep = iou_keep
        self.prev: tuple[int,int,int,int] | None = None

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

    def update(self, new_box: tuple[int,int,int,int] | None):
        if new_box is None:
            return self.prev
        if self.prev is None:
            self.prev = new_box
            return self.prev
        if self._iou(self.prev, new_box) < self.iou_keep:
            # 점프 → 즉시 치환
            self.prev = new_box
            return self.prev
        # EMA 스무딩
        px0, py0, px1, py1 = self.prev
        nx0, ny0, nx1, ny1 = new_box
        sx0 = int(self.alpha * nx0 + (1 - self.alpha) * px0)
        sy0 = int(self.alpha * ny0 + (1 - self.alpha) * py0)
        sx1 = int(self.alpha * nx1 + (1 - self.alpha) * px1)
        sy1 = int(self.alpha * ny1 + (1 - self.alpha) * py1)
        self.prev = (sx0, sy0, sx1, sy1)
        return self.prev

# =========================================================
#                     SHAPE FALLBACK
#   밝기+원형도 기반 후보 탐색 (YOLO 미검출 프레임 보조)
# =========================================================
class _ShapeFallback:
    def __init__(self, prev_expand=2.2, area_main=(1000, 2000), area_fb=(600, 4000)):
        self.prev_expand = prev_expand
        self.area_main = area_main
        self.area_fb = area_fb

    @staticmethod
    def _clip(x0, y0, x1, y1, w, h):
        x0 = max(0, min(x0, w - 1)); x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h))
        if x1 <= x0: x1 = x0 + 1
        if y1 <= y0: y1 = y0 + 1
        return x0, y0, x1, y1

    @staticmethod
    def _bbox_from_cnt(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, x + w, y + h)

    @staticmethod
    def _center(b):
        x0, y0, x1, y1 = b
        return ( (x0+x1)//2, (y0+y1)//2 )

    def detect(self, img_bgr, prev_bbox=None):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # 밝기 대역(208±10%)
        low, high = int(round(208 * 0.9)), int(round(208 * 1.1))
        mask = ((gray >= low) & (gray <= high)).astype(np.uint8) * 255

        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 31, -5)
        bw = cv2.bitwise_and(bw, mask)

        # 좌상/좌하/중앙 수평 밴드 + 이전 ROI 확장
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

        def _select(bin_img, area_rng):
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cand = []
            for c in contours:
                a = cv2.contourArea(c)
                if a < area_rng[0] or a > area_rng[1]:
                    continue
                peri = cv2.arcLength(c, True)
                if peri == 0: 
                    continue
                circ = 4.0 * np.pi * a / (peri * peri)
                if 0.55 <= circ <= 0.95:
                    cand.append(c)
            return cand

        cand1 = _select(bw_r, self.area_main)
        best = None
        if cand1:
            if prev_bbox is None:
                best = max(cand1, key=lambda c: cv2.contourArea(c))
            else:
                pcx, pcy = self._center(prev_bbox)
                def _negdist(c):
                    x0,y0,x1,y1 = self._bbox_from_cnt(c)
                    cx, cy = (x0+x1)//2, (y0+y1)//2
                    return -((cx-pcx)**2 + (cy-pcy)**2)
                best = max(cand1, key=_negdist)
        else:
            if prev_bbox is not None:
                px0, py0, px1, py1 = prev_bbox
                roi = np.zeros_like(bw_r); roi[py0:py1, px0:px1] = bw_r[py0:py1, px0:px1]
                cand2 = _select(roi, self.area_fb)
                target = cand2 if cand2 else _select(bw_r, self.area_fb)
                if target:
                    pcx, pcy = self._center(prev_bbox)
                    def _negdist(c):
                        x0,y0,x1,y1 = self._bbox_from_cnt(c)
                        cx, cy = (x0+x1)//2, (y0+y1)//2
                        return -((cx-pcx)**2 + (cy-pcy)**2)
                    best = max(target, key=_negdist)

        return None if best is None else self._bbox_from_cnt(best)

# =========================================================
#                       MASK UTILS
# =========================================================
def _bbox_to_mask(h: int, w: int, bbox: tuple[int,int,int,int] | None, feather=8) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        return mask
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
    mask[y0:y1, x0:x1] = 255
    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask

# =========================================================
#                   MAIN CLEANER (2.0)
# =========================================================
class WaterMarkCleaner:
    """
    KaraVision 2.0:
      - YOLO 탐지(외부 모듈) → Tracking → Shape Fallback → LaMa 인페인트
      - public API:
          * clean(input_image, watermark_mask)  # 호환용
          * clean_frame(frame_bgr)              # 영상 프레임 처리(권장)
    """
    def __init__(self, yolo_conf: float = 0.10, feather: int = 14, device: str | None = None):
        # LaMa 모델 준비 (기존 방식 재사용)
        self.model_name = DEFAULT_WATERMARK_REMOVE_MODEL
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        scanned = scan_models()
        if self.model_name not in [it.name for it in scanned]:
            logger.info(f"{self.model_name} not found in {DEFAULT_MODEL_DIR} (skip auto-download; use provided weights)")
        self.model_manager = ModelManager(name=self.model_name, device=self.device)
        self.inpaint_req = InpaintRequest()

        # Detector + Tracker + Fallback
        self.detector = SoraWaterMarkDetector()   # 기존 YOLO 래퍼 재사용
        self.tracker = _SimpleTracker(alpha=0.6, iou_keep=0.10)
        self.fallback = _ShapeFallback()
        self.yolo_conf = yolo_conf
        self.feather = feather

    # --- 1.x 호환용 시그니처(이미지+마스크 입력 시 인페인트만 수행) ---
    def clean(self, input_image: np.ndarray, watermark_mask: np.ndarray) -> np.ndarray:
        result = self.model_manager(input_image, watermark_mask, self.inpaint_req)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # --- 2.0 권장 API: 단일 프레임 처리 ---
    def clean_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        H, W = frame_bgr.shape[:2]

        # 1) YOLO 탐지
        det = self.detector.detect(frame_bgr)
        yolo_box = det["bbox"] if (det and det.get("detected", False)) else None

        # 2) 폴백(형태 기반) - YOLO 실패 시만
        pick = yolo_box if yolo_box is not None else self.fallback.detect(frame_bgr, self.tracker.prev)

        # 3) 트래킹/EMA
        smooth_box = self.tracker.update(pick)

        # 4) 마스크 생성 + 인페인트
       if smooth_box is None:
    # YOLO / Fallback / Tracker 모두 실패 → 원본 사용
       return frame_bgr

       mask = _bbox_to_mask(H, W, smooth_box, feather=self.feather)
       inpainted = self.model_manager(frame_bgr, mask, self.inpaint_req)
       return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
