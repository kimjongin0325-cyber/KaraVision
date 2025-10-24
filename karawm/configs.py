from pathlib import Path

# 프로젝트 루트 기준 경로
ROOT = Path(__file__).resolve().parents[1]

# 모델 저장 경로
MODEL_DIR = ROOT / "resources"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# YOLO 워터마크 탐지 모델 (best.pt)
WATER_MARK_DETECT_YOLO_WEIGHTS = MODEL_DIR / "best.pt"

# LaMa 인페인팅 모델 (big-lama.pt)
LAMA_WEIGHTS_PATH = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt"

# GitHub Release에서 자동 다운로드할 URL
MODEL_URLS = {
    "best.pt": "https://github.com/kimjongin0325-cyber/KaraVision/releases/download/v1.0/best.pt",
    "big-lama.pt": "https://github.com/kimjongin0325-cyber/KaraVision/releases/download/v1.0/big-lama.pt",
}

# 자동 다운로드 함수
def ensure_model_files():
    import requests

    # YOLO 모델
    if not WATER_MARK_DETECT_YOLO_WEIGHTS.exists():
        print(f"📥 Downloading best.pt from GitHub...")
        with requests.get(MODEL_URLS["best.pt"], stream=True) as r:
            r.raise_for_status()
            with open(WATER_MARK_DETECT_YOLO_WEIGHTS, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ best.pt downloaded.")

    # LaMa 모델
    if not LAMA_WEIGHTS_PATH.exists():
        print(f"📥 Downloading big-lama.pt from GitHub...")
        LAMA_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(MODEL_URLS["big-lama.pt"], stream=True) as r:
            r.raise_for_status()
            with open(LAMA_WEIGHTS_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ big-lama.pt downloaded.")

# 실행 시 자동 확인
ensure_model_files()
