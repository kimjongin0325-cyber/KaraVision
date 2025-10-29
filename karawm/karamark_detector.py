from loguru import logger
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


class KaramarkDetector:
    def __init__(self, conf=0.1, iou=0.45, device="cuda"):
        logger.debug("Begin to load YOLOv8m detector.")

        self.conf = conf
        self.iou = iou
        self.device = device

        # ✅ safe load override wrapper
        torch.serialization.add_safe_globals([DetectionModel])
        torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

        # ✅ A라인 전용: YOLOv8m.pt
        self.model = YOLO("resources/yolov8m.pt")
        logger.debug(f"YOLOv8m setup: conf={conf}, iou={iou}, device={device}")

    def detect(self, img_bgr):
        results = self.model(
            img_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False
        )

        dets = []
        for r in results:
            if hasattr(r, "boxes"):
                for box in r.boxes:
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    dets.append([*xyxy, conf, cls])

        return dets
