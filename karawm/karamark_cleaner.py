from loguru import logger
import numpy as np
import cv2

from .karamark_detector import KaramarkDetector
from .iopaint.inpaint_engine import InpaintEngine


class KaramarkCleaner:
    def __init__(self):
        self.detector = KaramarkDetector()
        self.inpainter = InpaintEngine()
        self.tracked_box = None
        self.frame_count = 0
        logger.info("✅ KaramarkCleaner ready (Detector + Inpainter)")

    def clean_frame(self, frame_bgr):
        self.frame_count += 1
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1].copy()

        # ✅ YOLO 감지
        dets = self.detector.detect(frame_bgr)

        # ✅ 감지 없으면 원본 유지
        if not dets:
            return frame_bgr

        # ✅ 점수 높은 박스 하나만 사용
        dets.sort(key=lambda d: d[4], reverse=True)
        x1, y1, x2, y2, conf, cls = dets[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # ✅ 위치 유효성 체크
        if x2 <= x1 or y2 <= y1:
            return frame_bgr

        # ✅ Tracking (초간단)
        if self.frame_count <= 3:  # 앞 3프레임은 YOLO
            self.tracked_box = (x1, y1, x2, y2)
        else:  # 이후에는 추정 유지
            if self.tracked_box is not None:
                x1, y1, x2, y2 = self.tracked_box

        # ✅ 인페인트 영역 확장
        pad = 10
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(frame_w, x2 + pad)
        y2p = min(frame_h, y2 + pad)

        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask[y1p:y2p, x1p:x2p] = 255

        # ✅ 인페인트
        result = self.inpainter(frame_rgb, mask)

        return result[:, :, ::-1]
