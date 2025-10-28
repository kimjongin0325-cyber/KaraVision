from loguru import logger
import numpy as np
import torch
import joblib
import cv2

from .karamark_detector import KaramarkDetector
from .iopaint.inpaint_engine import InpaintEngine


class KaramarkCleaner:
    def __init__(self):
        self.detector = KaramarkDetector()
        self.inpainter = InpaintEngine()
        self.classifier = joblib.load("resources/logo_classifier.pkl")  # ✅ 로고 분류기 로드
        logger.info("✅ KaramarkCleaner ready (Detector + Classifier + Inpainter)")

    def clean_frame(self, frame_bgr):
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1].copy()  # ✅ BGR → RGB

        dets = self.detector.detect(frame_bgr)

# ✅ Detection 없으면 원본 유지
if not dets:
    return frame_bgr

# ✅ Multi-logo support
dets.sort(key=lambda d: d[4], reverse=True)

result = frame_rgb.copy()  # 여러 번 inpaint 적용 대비

for det in dets:
    x1, y1, x2, y2, conf, cls = det
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # ✅ 범위 검증
    if x2 <= x1 or y2 <= y1:
        continue

    # ✅ 분류기 체크 (진짜 로고인지 확인)
    crop = result[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    crop_resized = cv2.resize(crop, (64, 64))
    crop_flat = crop_resized.flatten().reshape(1, -1)
    pred = self.classifier.predict_proba(crop_flat)[0][1]

    if pred < 0.5:
        logger.debug(f"❌ Not a logo: pred={pred:.2f}")
        continue

    # ✅ 주변까지 넓게 제거 (품질↑)
    pad = 10
    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
    x2p, y2p = min(frame_w, x2 + pad), min(frame_h, y2 + pad)

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    mask[y1p:y2p, x1p:x2p] = 255

    result = self.inpainter(result, mask)

# ✅ 최종 BGR
return result[:, :, ::-1]
