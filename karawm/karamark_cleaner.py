from loguru import logger
import numpy as np
import torch

from .karamark_detector import KaramarkDetector
from .iopaint.inpaint_engine import InpaintEngine


class KaramarkCleaner:
    def __init__(self):
        self.detector = KaramarkDetector()
        self.inpainter = InpaintEngine()
        logger.info("✅ KaramarkCleaner ready!")

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

        # ✅ mask 생성
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        # ✅ Inpaint 실행
        result_rgb = self.inpainter(frame_rgb, mask)

        # ✅ Tensor 처리 대비
        if isinstance(result_rgb, torch.Tensor):
            result_rgb = result_rgb.cpu().numpy()

        # ✅ 채널 보정
        if result_rgb.ndim == 2:
            result_rgb = np.stack([result_rgb] * 3, axis=-1)
        elif result_rgb.shape[2] == 1:
            result_rgb = np.repeat(result_rgb, 3, axis=2)

        # ✅ 다시 BGR 변환
        return result_rgb[:, :, ::-1]
