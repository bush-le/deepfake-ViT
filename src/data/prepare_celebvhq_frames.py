"""
Prepare CelebV-HQ real frames for v5 weak-method finetune.
Extracts N evenly-spaced frames from a deterministic subset of celebvhq videos
into /workspace/data/celebvhq_frames/real/*.jpg (512x512, used as REAL source).
Deterministic (seed) so runs are reproducible.
"""
import os, sys, glob, cv2, random, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DF40_ROOT", PROJECT_ROOT / "data"))

VIDEO_DIR = DATA_ROOT / "celebvhq/35666"
OUT_DIR   = DATA_ROOT / "celebvhq_frames/real"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-videos", type=int, default=200)
    ap.add_argument("--frames-per-video", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = sorted(glob.glob(str(VIDEO_DIR / "*.mp4")))
    print(f"Total videos available: {len(videos)}")
    rng = random.Random(args.seed)
    selected = rng.sample(videos, min(args.n_videos, len(videos)))
    print(f"Extracting from {len(selected)} videos, {args.frames_per_video} frames each")

    n_written = 0
    for vpath in selected:
        vid = Path(vpath).stem
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            cap.release(); continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release(); continue
        # evenly spaced indices (include last frame)
        idxs = sorted(set(round(i * (total - 1) / (args.frames_per_video - 1))
                          for i in range(args.frames_per_video)))
        for fidx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, fr = cap.read()
            if not ok:
                continue
            out = OUT_DIR / f"{vid}_{fidx:04d}.jpg"
            cv2.imwrite(str(out), fr)
            n_written += 1
        cap.release()

    print(f"Wrote {n_written} real frames to {OUT_DIR}")

if __name__ == "__main__":
    main()
