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
        self.classifier = joblib.load("resources/logo_classifier.pkl")
        self.tracked_box = None       # ✅ Tracking 좌표
        self.frame_count = 0          # ✅ Frame index
        logger.info("✅ KaramarkCleaner ready (Detector + Classifier + Inpainter)")

    def clean_frame(self, frame_bgr):
        self.frame_count += 1
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1].copy()

        # ✅ YOLO 탐지
        dets = self.detector.detect(frame_bgr)

        # ✅ Tracking 적용 (첫 3프레임만 YOLO 결과 반영)
        tracking_frames = 3
        if self.frame_count <= tracking_frames:
            if dets:
                dets.sort(key=lambda d: d[4], reverse=True)
                self.tracked_box = dets[0][:4]  # (x1,y1,x2,y2)
        else:
            if self.tracked_box is not None:
                x1, y1, x2, y2 = map(int, self.tracked_box)
                dets = [[x1, y1, x2, y2, 1.0, -1]]
            else:
                return frame_bgr

        # ✅ Detection 없으면 원본 유지
        if not dets:
            return frame_bgr

        # ✅ Multi-logo support
        dets.sort(key=lambda d: d[4], reverse=True)
        result = frame_rgb.copy()

        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # ✅ ROI 유효성
            if x2 <= x1 or y2 <= y1:
                continue

            # ✅ 분류기로 진짜 로고인지 확인
            crop = result[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (64, 64))
            crop_flat = crop_resized.flatten().reshape(1, -1)
            pred = self.classifier.predict_proba(crop_flat)[0][1]

            if pred < 0.5:
                logger.debug(f"❌ Not a logo: pred={pred:.2f}")
                continue
            logger.debug(f"✅ Logo confirmed: pred={pred:.2f}")

            # ✅ 인페인트 영역 확장 (품질↑)
            pad = 10
            x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
            x2p, y2p = min(frame_w, x2 + pad), min(frame_h, y2 + pad)

            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            mask[y1p:y2p, x1p:x2p] = 255

            result = self.inpainter(result, mask)

        return result[:, :, ::-1]
