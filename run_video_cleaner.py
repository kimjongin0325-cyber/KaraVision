from pathlib import Path
import argparse
import cv2
from karawm.karamark_cleaner import KaramarkCleaner  # ✅ 정확한 클래스명

def main(input_path, output_path):
    cleaner = KaramarkCleaner()  # ✅ 인스턴스 생성도 일치

    cap = cv2.VideoCapture(str(input_path))
    assert cap.isOpened(), f"입력 영상 열기 실패: {input_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ✅ Colab 권장 코덱 설정
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    frame_idx = 0
    print("[INFO] 처리 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = cleaner.clean_frame(frame)
        writer.write(result)
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"[완료] 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(Path(args.input), Path(args.output))
