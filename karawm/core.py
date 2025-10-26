from pathlib import Path
from typing import Callable

import ffmpeg
import numpy as np
from loguru import logger
from tqdm import tqdm

from karawm.utils.video_utils import VideoLoader
from karawm.karamk_cleaner import KaramkCleaner
from karawm.karamk_detector import KaramkDetector


class karawm:
    def __init__(self):
        self.detector = KaramkDetector()
        self.cleaner = KaramkCleaner()

    def run(
        self,
        input_video_path: Path,
        output_video_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ):
        input_video_loader = VideoLoader(input_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        width = input_video_loader.width
        height = input_video_loader.height
        fps = input_video_loader.fps
        total_frames = input_video_loader.total_frames
        # Create temporary output path for video without audio
        temp_output_path = output_video_path.parent / f"temp_{output_video_path.name}"

        process_out = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="bgr24",
                s=f"{width}x{height}",
                r=fps,
            )
            .output(str(temp_output_path), pix_fmt="yuv420p", vcodec="libx264")
            .overwrite_output()
            .global_args("-loglevel", "error")
            .run_async(pipe_stdin=True)
        )

        frame_and_mask = {}
        detect_missed = []

        logger.debug(
            f"total frames: {total_frames}, fps: {fps}, width: {width}, height: {height}"
        )
        for idx, frame in enumerate(
            tqdm(input_video_loader, total=total_frames, desc="Detect karamks")
        ):
            detection_result = self.detector.detect(frame)
            if detection_result["detected"]:
                frame_and_mask[idx] = {"frame": frame, "bbox": detection_result["bbox"]}
            else:
                frame_and_mask[idx] = {"frame": frame, "bbox": None}
                detect_missed.append(idx)

            # 10% - 50%
            if progress_callback and idx % 10 == 0:
                progress = 10 + int((idx / total_frames) * 40)
                progress_callback(progress)

        logger.debug(f"detect missed frames: {detect_missed}")

        for missed_idx in detect_missed:
            before = max(missed_idx - 1, 0)
            after = min(missed_idx + 1, total_frames - 1)
            before_box = frame_and_mask[before]["bbox"]
            after_box = frame_and_mask[after]["bbox"]
            if before_box:
                frame_and_mask[missed_idx]["bbox"] = before_box
            elif after_box:
                frame_and_mask[missed_idx]["bbox"] = after_box

        for idx in tqdm(range(total_frames), desc="Remove karamks"):
            frame_info = frame_and_mask[idx]
            frame = frame_info["frame"]
            bbox = frame_info["bbox"]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                mask = np.zeros((height, width), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255
                cleaned_frame = self.cleaner.clean(frame, mask)
            else:
                cleaned_frame = frame
            process_out.stdin.write(cleaned_frame.tobytes())

            # 50% - 95%
            if progress_callback and idx % 10 == 0:
                progress = 50 + int((idx / total_frames) * 45)
                progress_callback(progress)

        process_out.stdin.close()
        process_out.wait()

        # 95% - 99%
        if progress_callback:
            progress_callback(95)

        self.merge_audio_track(input_video_path, temp_output_path, output_video_path)

        if progress_callback:
            progress_callback(99)

    def merge_audio_track(
        self, input_video_path: Path, temp_output_path: Path, output_video_path: Path
    ):
        logger.info("Merging audio track...")
        video_stream = ffmpeg.input(str(temp_output_path))
        audio_stream = ffmpeg.input(str(input_video_path)).audio

        (
            ffmpeg.output(
                video_stream,
                audio_stream,
                str(output_video_path),
                vcodec="copy",
                acodec="aac",
            )
            .overwrite_output()
            .run(quiet=True)
        )
        # Clean up temporary file
        temp_output_path.unlink()
        logger.info(f"Saved no karamk video with audio at: {output_video_path}")


if __name__ == "__main__":
    from pathlib import Path

    input_video_path = Path(
        "resources/19700121_1645_68e0a027836c8191a50bea3717ea7485.mp4"
    )
    output_video_path = Path("outputs/sora_karamk_removed.mp4")
    sora_wm = karawm()
    sora_wm.run(input_video_path, output_video_path)
