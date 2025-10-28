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
        frame_rgb = frame_bgr[:,from loguru import logger
import numpy as np
import torch

from .karamark_detector import KaramarkDetector  # ✅ 중요!
from .iopaint.inpaint_engine import InpaintEngine  # ✅ LaMa 엔진

class KaramarkCleaner:
    def __init__(self):
        self.detector = KaramarkDetector()
        self.inpainter = InpaintEngine()
        logger.info("✅ KaramarkCleaner ready!")


    def clean_frame(self, frame_bgr):
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1].copy()  # ✅ stride fix, BGR→RGB

        det = self.detector.detect(frame_bgr)

        # ✅ YOLOv8 Detection 처리
        if det and len(det) > 0:
            x1, y1, x2, y2, conf, cls = det[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        else:
            return frame_bgr  # YOLO 못 찾으면 원본 그대로

        # ✅ 마스크 만들기
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # ✅ Inpaint 엔진 실행
        result_rgb = self.inpainter(frame_rgb, mask)

        # ✅ 채널 검사 (LaMa가 1채널일 경우 3채널 복구)
        if result_rgb.ndim == 2:  # (H, W)
            result_rgb = np.stack([result_rgb] * 3, axis=-1)
        elif result_rgb.shape[2] == 1:
            result_rgb = np.repeat(result_rgb, 3, axis=2)

        # ✅ 최종 출력 다시 BGR로 변환
        return result_rgb[:, :, ::-1]
 :, ::-1]  # BGR → RGB

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
      
