from loguru import logger
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


class KaramarkDetector:
    def __init__(self, conf=0.1, iou=0.45, device="cuda"):
        logger.debug("Begin to load YOLOv8m + best.pt detector.")

        self.conf = conf
        self.iou = iou
        self.device = device

        torch.serialization.add_safe_globals([DetectionModel])
        torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])

        # Stage1: 넓게 탐지
        self.model_primary = YOLO("resources/yolov8m.pt")

        # Stage2: 정밀 필터링
        self.model_refine = YOLO("resources/best.pt")

        logger.debug(f"YOLOv8m + best.pt setup: conf={conf}, iou={iou}, device={device}")

    def detect(self, img_bgr):
        # Stage1 후보 추출
        results = self.model_primary(
            img_bgr, conf=self.conf, iou=self.iou,
            device=self.device, verbose=False
        )

        candidates = []
        for r in results:
            if not hasattr(r, "boxes"): continue
            for b in r.boxes:
                xyxy = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                candidates.append([*xyxy, conf, cls])

        if not candidates:
            return []

        # Stage2 best.pt로 정밀 확인
        refined = []
        for det in candidates:
            x1, y1, x2, y2, conf, cls = det
            x1, y1 = int(x1), int(y1)
            x2, y2 = int(x2), int(y2)

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            r2 = self.model_refine(
                crop, conf=0.25, iou=0.4,
                device=self.device, verbose=False
            )

            if hasattr(r2[0], "boxes") and len(r2[0].boxes) > 0:
                refined.append(det)

        return refined
