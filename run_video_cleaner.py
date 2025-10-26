from pathlib import Path
import argparse
import cv2
from karawm.watermark_cleaner import WaterMarkCleaner

def main(input_path, output_path):
    cleaner = WaterMarkCleaner()

    cap = cv2.VideoCapture(str(input_path))
    assert cap.isOpened(), f"입력 영상 열기 실패: {input_path}"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"avc1"),
        fps,
        (w, h),
    )

    frame_idx = 0
    print("[INFO] 처리 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 👉 물리적으로는 탐지 + 제거 + 인페인트
        result = cleaner.clean_frame(frame)
        writer.write(result)

        frame_idx += 1
        cv2.imshow("Preview", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[완료] 저장: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(Path(args.input), Path(args.output))
