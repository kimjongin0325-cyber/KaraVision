from loguru import logger
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


class KaramarkDetector:
    def __init__(self, conf=0.1, iou=0.45, device="cuda"):
        logger.debug("Begin to load yolo detect model.")

        self.conf = conf
        self.iou = iou
        self.device = device

        # ✅ safe load override wrapper
        torch.serialization.add_safe_globals([DetectionModel])
        torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

        # ✅ best.ckpt: 로고 특화 탐지 모델
        self.model = YOLO("resources/best.ckpt")
        logger.debug(f"YOLO setup: conf={conf}, iou={iou}, device={device}")

    def detect(self, img_bgr):
        results = self.model(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device
        )

        dets = []
        for r in results:
            if hasattr(r, "boxes"):
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                    cls = int(box.cls[0]) if hasattr(box, "cls") else -1
                    dets.append([*xyxy, conf, cls])

        return dets
