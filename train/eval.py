from pathlib import Path

import numpy as np
from ultralytics import YOLO

from karawm.configs import KARA_MARK_DETECT_YOLO_WEIGHTS
from karawm.utils.video_utils import VideoLoader

# based on the kara tempalte to detect the whole, and then got the icon part area.

model = YOLO(KARA_MARK_DETECT_YOLO_WEIGHTS)
model.eval()

results = model("resources/first_frame.png")  # Predict on an image

print(results)
