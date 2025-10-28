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

        # ✅ 가장 높은 confidence 사용
        dets.sort(key=lambda d: d[4], reverse=True)
        x1, y1, x2, y2, conf, cls = dets[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # ✅ 분류기: 진짜 로고인지 확인
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return frame_bgr

        crop_resized = cv2.resize(crop, (64, 64))
        crop_flat = crop_resized.flatten().reshape(1, -1)
        pred = self.classifier.predict_proba(crop_flat)[0][1]

        if pred < 0.5:
            logger.debug(f"❌ Not a logo: pred={pred:.2f}")
            return frame_bgr
        logger.debug(f"✅ Logo detected: pred={pred:.2f}")

        # ✅ 주변까지 넓게 제거 (품질 향상)
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame_w, x2 + pad), min(frame_h, y2 + pad)

        # ✅ mask 생성
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # ✅ Inpaint 실행
        result_rgb = self.inpainter(frame_rgb, mask)

        # ✅ Tensor → NumPy
        if isinstance(result_rgb, torch.Tensor):
            result_rgb = result_rgb.cpu().numpy()

        # ✅ 채널 보정
        if result_rgb.ndim == 2:
            result_rgb = np.stack([result_rgb] * 3, axis=-1)
        elif result_rgb.shape[2] == 1:
            result_rgb = np.repeat(result_rgb, 3, axis=2)

        # ✅ 최종 BGR 반환
        return result_rgb[:, :, ::-1]
