from loguru import logger
import numpy as np

from .karamark_detector import KaramarkDetector
from .iopaint.inpaint_engine import InpaintEngine


class KaramarkCleaner:
    def __init__(self):
        self.detector = KaramarkDetector()
        self.inpainter = InpaintEngine()
        logger.info("✅ KaramarkCleaner ready!")

    def clean_frame(self, frame_bgr):
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR → RGB

        det = self.detector.detect(frame_bgr)

        # ✅ YOLOv8 결과 처리
        if det and len(det) > 0:
            x1, y1, x2, y2, conf, cls = det[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        else:
            return frame_bgr  # YOLO 못 찾으면 원본 유지

        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255  # ✅ bounding box 영역만 덮기

        # ✅ 인페인트 실행
      
