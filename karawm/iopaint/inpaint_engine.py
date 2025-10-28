import torch


class InpaintEngine:
    def __call__(self, frame_rgb, mask):
        frame_tensor = torch.from_numpy(frame_rgb.copy()).float().cuda() / 255.0
        mask_tensor = torch.from_numpy(mask.copy()).float().cuda() / 255.0

        # ✅ 간단한 평균색 덮기
        frame_tensor[mask_tensor > 0] = frame_tensor.mean()

        result = (frame_tensor.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        return result
