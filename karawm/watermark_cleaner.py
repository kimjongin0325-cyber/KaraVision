from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from loguru import logger

# === (1) IOPaint / LaMa ===
from karawm.iopaint.model_manager import ModelManager
from karawm.iopaint.schema import InpaintRequest
from karawm.configs import DEFAULT_WATERMARK_REMOVE_MODEL

# === (2) YOLO ===
# pip install ultralytics 필요 (최초 1회)
try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit(
        "Ultralytics가 필요합니다. 먼저 `pip install ultralytics` 실행 후 다시 시도하세요."
    )

# === (3) 모델 파일 확보 ===
# 네가 올려둔 ensure_model_files() 재사용
# - YOLO: resources/best.pt
# - LaMa: ~/.cache/torch/hub/checkpoints/big-lama.pt
from pathlib import Path as _P

ROOT = _P(__file__).resolve().parent
MODEL_DIR = ROOT / "resources"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WATER_MARK_DETECT_YOLO_WEIGHTS = MODEL_DIR / "best.pt"
LAMA_WEIGHTS_PATH = _P.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt"

MODEL_URLS = {
    "best.pt": "https://github.com/kimjongin0325-cyber/KaraVision/releases/download/v1.0/best.pt",
    "big-lama.pt": "https://github.com/kimjongin0325-cyber/KaraVision/releases/download/v1.0/big-lama.pt",
}

def ensure_model_files():
    import requests
    if not WATER_MARK_DETECT_YOLO_WEIGHTS.exists():
        print("📥 Downloading best.pt from GitHub...")
        with requests.get(MODEL_URLS["best.pt"], stream=True) as r:
            r.raise_for_status()
            with open(WATER_MARK_DETECT_YOLO_WEIGHTS, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ best.pt downloaded.")
    if not LAMA_WEIGHTS_PATH.exists():
        print("📥 Downloading big-lama.pt from GitHub...")
        LAMA_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(MODEL_URLS["big-lama.pt"], stream=True) as r:
            r.raise_for_status()
            with open(LAMA_WEIGHTS_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ big-lama.pt downloaded.")

# =========================================================
#                       DETECTORS
# =========================================================
class YoloDetector:
    def __init__(self, weights_path: Path, conf_thres=0.25, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(str(weights_path))
        self.conf = conf_thres

    def __call__(self, image_bgr):
        """
        Returns: list of (x0, y0, x1, y1, conf)
        """
        results = self.model.predict(source=image_bgr, verbose=False, conf=self.conf, device=0 if self.device=="cuda" else None)
        boxes = []
        if not results:
            return boxes
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return boxes
        for b in r0.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)  # [x1,y1,x2,y2]
            conf = float(b.conf[0].cpu().numpy())
            x0, y0, x1, y1 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            boxes.append((x0, y0, x1, y1, conf))
        return boxes

# --------- C: 형태 기반 (밝기/원형도) + ROI 트래킹 보조 ----------
class ShapeFallbackDetector:
    def __init__(self, prev_expand=2.2, area_main=(1000, 2000), area_fallback=(600, 4000)):
        self.prev_expand = prev_expand
        self.area_main = area_main
        self.area_fallback = area_fallback

    @staticmethod
    def _clip_rect(x0, y0, x1, y1, w_img, h_img):
        x0 = max(0, min(x0, w_img - 1))
        x1 = max(0, min(x1, w_img))
        y0 = max(0, min(y0, h_img - 1))
        y1 = max(0, min(y1, h_img))
        if x1 <= x0: x1 = x0 + 1
        if y1 <= y0: y1 = y0 + 1
        return x0, y0, x1, y1

    @staticmethod
    def _cnt_bbox(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, x + w, y + h)

    @staticmethod
    def _center(b):
        x0, y0, x1, y1 = b
        return ((x0 + x1)//2, (y0 + y1)//2)

    def detect(self, image_bgr, prev_bbox=None):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        low, high = int(round(208*0.9)), int(round(208*1.1))
        mask = ((gray >= low) & (gray <= high)).astype(np.uint8) * 255
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, -5)
        bw = cv2.bitwise_and(bw, mask)

        h, w = gray.shape[:2]
        r_top_left = (0, 0, int(0.2*w), int(0.2*h))
        r_bot_left = (0, int(0.8*h), int(0.2*w), h)
        y0, y1 = int(0.40*h), int(0.60*h)
        r_mid_band = (0, y0, w, y1)

        region_mask = np.zeros_like(bw, dtype=np.uint8)
        for x0, ys, x1, ye in (r_top_left, r_bot_left):
            region_mask[ys:ye, x0:x1] = 255
        region_mask[y0:y1, :] = 255

        prev_roi = None
        if prev_bbox is not None:
            px0, py0, px1, py1 = prev_bbox
            pw, ph = (px1 - px0), (py1 - py0)
            cx, cy = self._center(prev_bbox)
            rw, rh = int(pw * self.prev_expand), int(ph * self.prev_expand)
            rx0, ry0 = cx - rw//2, cy - rh//2
            rx1, ry1 = cx + rw//2, cy + rh//2
            rx0, ry0, rx1, ry1 = self._clip_rect(rx0, ry0, rx1, ry1, w, h)
            region_mask[ry0:ry1, rx0:rx1] = 255
            prev_roi = (rx0, ry0, rx1, ry1)

        bw_region = cv2.bitwise_and(bw, region_mask)

        def select_candidates(bw_bin, area_rng):
            contours, _ = cv2.findContours(bw_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cand = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < area_rng[0] or area > area_rng[1]:
                    continue
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circularity = 4.0*np.pi*area/(peri*peri)
                if 0.55 <= circularity <= 0.95:
                    cand.append(cnt)
            return contours, cand

        contours, cand1 = select_candidates(bw_region, self.area_main)
        best_bbox = None
        if cand1:
            if prev_bbox is None:
                best_cnt = max(cand1, key=lambda c: cv2.contourArea(c))
                best_bbox = self._cnt_bbox(best_cnt)
            else:
                pcx, pcy = self._center(prev_bbox)
                def neg_dist(c):
                    x0,y0,x1,y1 = self._cnt_bbox(c)
                    cx, cy = (x0+x1)//2, (y0+y1)//2
                    return -((cx-pcx)**2 + (cy-pcy)**2)
                best_cnt = max(cand1, key=neg_dist)
                best_bbox = self._cnt_bbox(best_cnt)
        else:
            if prev_roi is not None:
                rx0, ry0, rx1, ry1 = prev_roi
                roi = np.zeros_like(bw_region)
                roi[ry0:ry1, rx0:rx1] = bw_region[ry0:ry1, rx0:rx1]
                _, cand2 = select_candidates(roi, self.area_fallback)
                if cand2:
                    if prev_bbox is None:
                        best_cnt = max(cand2, key=lambda c: cv2.contourArea(c))
                    else:
                        pcx, pcy = self._center(prev_bbox)
                        def neg_dist(c):
                            x0,y0,x1,y1 = self._cnt_bbox(c)
                            cx, cy = (x0+x1)//2, (y0+y1)//2
                            return -((cx-pcx)**2 + (cy-pcy)**2)
                        best_cnt = max(cand2, key=neg_dist)
                    best_bbox = self._cnt_bbox(best_cnt)
                else:
                    if prev_bbox is not None:
                        _, cand3 = select_candidates(bw_region, self.area_fallback)
                        if cand3:
                            pcx, pcy = self._center(prev_bbox)
                            def neg_dist(c):
                                x0,y0,x1,y1 = self._cnt_bbox(c)
                                cx, cy = (x0+x1)//2, (y0+y1)//2
                                return -((cx-pcx)**2 + (cy-pcy)**2)
                            best_cnt = max(cand3, key=neg_dist)
                            best_bbox = self._cnt_bbox(best_cnt)

        return best_bbox  # or None

# =========================================================
#                      TRACKING / SMOOTH
# =========================================================
class SimpleTracker:
    """
    EMA 스무딩 + IoU 기반 갱신. (OpenCV tracker 불필요)
    """
    def __init__(self, alpha=0.6, iou_keep=0.1):
        self.alpha = alpha
        self.iou_keep = iou_keep
        self.prev = None

    @staticmethod
    def iou(a, b):
        if a is None or b is None: return 0.0
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        xi0, yi0 = max(ax0,bx0), max(ay0,by0)
        xi1, yi1 = min(ax1,bx1), min(ay1,by1)
        w, h = max(0, xi1-xi0), max(0, yi1-yi0)
        inter = w*h
        area_a = max(0, ax1-ax0)*max(0, ay1-ay0)
        area_b = max(0, bx1-bx0)*max(0, by1-by0)
        union = area_a + area_b - inter + 1e-6
        return inter/union

    def update(self, new_box):
        if new_box is None:
            return self.prev  # 유지
        if self.prev is None:
            self.prev = new_box
            return self.prev
        if self.iou(self.prev, new_box) < self.iou_keep:
            # 점프 → 바로 치환
            self.prev = new_box
            return self.prev
        # EMA 스무딩
        px0, py0, px1, py1 = self.prev
        nx0, ny0, nx1, ny1 = new_box
        sx0 = int(self.alpha*nx0 + (1-self.alpha)*px0)
        sy0 = int(self.alpha*ny0 + (1-self.alpha)*py0)
        sx1 = int(self.alpha*nx1 + (1-self.alpha)*px1)
        sy1 = int(self.alpha*ny1 + (1-self.alpha)*py1)
        self.prev = (sx0, sy0, sx1, sy1)
        return self.prev

# =========================================================
#                         MASK UTILS
# =========================================================
def bbox_to_mask(h, w, bbox, feather=6):
    """
    bbox -> (H,W) 0/255 mask with feathered edges
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if bbox is None:
        return mask
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1); y1 = min(h, y1)
    mask[y0:y1, x0:x1] = 255
    if feather > 0:
        k = feather if feather % 2 == 1 else feather + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        # 스무딩 후 0~255 유지
    return mask

# =========================================================
#                         CLEANER
# =========================================================
class KaraVisionCleaner:
    """
    D: YOLO 기반 탐지 → 우선
    C: 형태기반 Fallback
    B: 트래킹/EMA 스무딩
    """
    def __init__(self, yolo_weights: Path, lama_name: str = DEFAULT_WATERMARK_REMOVE_MODEL,
                 yolo_conf=0.25, feather=6, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.yolo = YoloDetector(yolo_weights, conf_thres=yolo_conf, device=str(self.device))
        self.fallback = ShapeFallbackDetector()
        self.tracker = SimpleTracker(alpha=0.6, iou_keep=0.1)
        self.feather = feather

        self.model_manager = ModelManager(name=lama_name, device=self.device)
        self.inpaint_req = InpaintRequest()

    def _choose_yolo_box(self, boxes, prev_box):
        if not boxes:
            return None
        # prev 중심에 가까운 박스 우선, 없으면 최고 conf
        if prev_box is None:
            return max(boxes, key=lambda b: b[4])[:4]
        pcx = (prev_box[0] + prev_box[2]) // 2
        pcy = (prev_box[1] + prev_box[3]) // 2
        def neg_dist(b):
            x0,y0,x1,y1,conf = b
            cx, cy = (x0+x1)//2, (y0+y1)//2
            return -((cx-pcx)**2 + (cy-pcy)**2)
        return max(boxes, key=neg_dist)[:4]

    def process_frame(self, frame_bgr):
        H, W = frame_bgr.shape[:2]

        # 1) YOLO
        yolo_boxes = self.yolo(frame_bgr)
        prev_box = self.tracker.prev
        yolo_pick = self._choose_yolo_box(yolo_boxes, prev_box)

        # 2) Fallback(C) if YOLO fail
        if yolo_pick is None:
            shape_box = self.fallback.detect(frame_bgr, prev_box)
            pick = shape_box
        else:
            pick = yolo_pick

        # 3) Tracking/EMA(B)
        smooth_box = self.tracker.update(pick)

        # 4) Mask + Inpaint
        mask = bbox_to_mask(H, W, smooth_box, feather=self.feather)
        inpainted = self.model_manager(frame_bgr, mask, self.inpaint_req)
        inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)  # IOPaint가 BGR 반환 → RGB 정규화

        return inpainted, smooth_box, (yolo_pick is not None)

# =========================================================
#                         RUN LOOP
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="입력 비디오 경로")
    parser.add_argument("--output", type=str, default="outputs/cleaned.mp4", help="출력 비디오 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence")
    parser.add_argument("--feather", type=int, default=6, help="mask feather(px, 가우시안)")
    parser.add_argument("--show", action="store_true", help="화면 미리보기")
    parser.add_argument("--fps", type=float, default=0.0, help="강제 FPS (0=원본 유지)")
    args = parser.parse_args()

    ensure_model_files()

    cap = cv2.VideoCapture(args.input)
    assert cap.isOpened(), f"Cannot open video: {args.input}"
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = args.fps if args.fps > 0 else src_fps

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    assert writer.isOpened(), "출력 비디오 생성 실패"

    cleaner = KaraVisionCleaner(
        yolo_weights=WATER_MARK_DETECT_YOLO_WEIGHTS,
        yolo_conf=args.conf,
        feather=args.feather
    )

    window = "KaraVision 2.0 - Watermark Cleaner (YOLO + Shape Fallback + Tracking)"
    if args.show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    idx = 0
    yolo_hits = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cleaned, bbox, used_yolo = cleaner.process_frame(frame)
        yolo_hits += int(used_yolo)

        # overlay (디버그)
        vis = cleaned.copy()
        if bbox is not None:
            x0,y0,x1,y1 = bbox
            cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)
            cv2.putText(vis, f"YOLO:{'Y' if used_yolo else 'N'}", (x0, max(0,y0-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        writer.write(vis)
        if args.show:
            cv2.imshow(window, vis)
            key = cv2.waitKey(int(1000/max(1,int(fps)))) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                while True:
                    k = cv2.waitKey(30) & 0xFF
                    if k in (ord(' '), ord('q')):
                        break
                if k == ord('q'):
                    break

        idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    logger.info(f"✅ Done. saved -> {out_path}")
    logger.info(f"YOLO hit ratio: {yolo_hits}/{max(1, idx)} = {yolo_hits/max(1,idx):.2%}")

if __name__ == "__main__":
    main()
