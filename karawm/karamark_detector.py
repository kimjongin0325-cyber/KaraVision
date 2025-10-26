from pathlib import Path

import numpy as np
import torch
from loguru import logger
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


class KaramarkDetector:
    def __init__(self, conf=0.25, iou=0.45, device=None):
        logger.debug("Begin to load yolo detect model.")

        import torch.serialization
        # ✅ PyTorch 2.6 안전모드에서 YOLO 모델 허용
        torch.serialization.add_safe_globals([DetectionModel])
        torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

        # ✅ safe load override wrapper
        def load_yolo_with_force(weights="yolov8n.pt"):
            try:
                return YOLO(weights)  # weights_only=True 실패하면 except로
            except Exception:
                logger.warning("Safe-load failed. Trying full checkpoint load...")
                ckpt = torch.load(weights, map_location="cpu", weights_only=False)
                model = YOLO()
                model.model = ckpt["model"]
                return model

        # ✅ 강제 로드 적용
        self.model = load_yolo_with_force("yolov8n.pt")

        self.model.conf = conf
        self.model.iou = iou

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        logger.debug(f"YOLO setup: conf={conf}, iou={iou}, device={self.device}")
