from pathlib import Path

from karawm.core import karawm

if __name__ == "__main__":
    input_video_path = Path("resources/dog_vs_sam.mp4")
    output_video_path = Path("outputs/sora_karamk_removed.mp4")
    sora_wm = karawm()
    sora_wm.run(input_video_path, output_video_path)
