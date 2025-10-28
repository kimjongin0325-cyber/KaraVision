import torch
import cv2
import numpy as np


class InpaintEngine:
    def __init__(self, model_path="/content/KaraVision/resources/big-lama.pt"):
        print("✅ Loading LaMa model for inpainting...")
        self.model = torch.jit.load(model_path, map_location="cuda").eval()

    def __call__(self, frame_rgb, mask):
        # ✅ BGR→RGB 이미 frame_rgb가 RGB임
        img = frame_rgb.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        # ✅ 모델 입력 텐서 변환
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).cuda()
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda()

        with torch.no_grad():
            result = self.model(img_t, mask_t)[0].cpu().numpy()

        # ✅ Tensor -> uint8 BGR 변환 (for OpenCV)
        result = (result.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

        return result

