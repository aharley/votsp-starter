"""CoTracker3 wrapper for VOT folder protocol.

Reads frames_color.txt and query_*.txt from the current directory,
runs CoTracker3 to track each queried point, and writes output_*.txt.

Usage (via trackers.ini):
    [CoTracker3]
    command = cotracker_tracker
    protocol = folderpython
    paths = /Users/aharley/votp
"""

import os
import numpy as np
import cv2
import torch

from vot.region.io import parse_region
from vot.region import Point

# CoTracker internally rescales all input to 384x512 regardless of this setting,
# so changing MAX_SIZE does not affect tracking quality — only memory for the raw tensor.
MAX_SIZE = 512


def load_model():
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", pretrained=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("CoTracker3 loaded.")
    return model


def load_frames(paths):
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise RuntimeError(f"Could not read: {p}")
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def resize_frames(imgs, max_size):
    H0, W0 = imgs[0].shape[:2]
    scale = min(max_size / H0, max_size / W0, 1.0)
    H = max(8, (int(H0 * scale) // 8) * 8)
    W = max(8, (int(W0 * scale) // 8) * 8)
    resized = [cv2.resize(img, (W, H)) for img in imgs]
    return resized, H, W, W / W0, H / H0


def to_tensor(imgs):
    arr = np.stack(imgs, axis=0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0)  # (1, T, 3, H, W)


def main():
    frame_files = sorted(f for f in os.listdir(".") if f.startswith("frames_") and f.endswith(".txt"))
    assert frame_files, "No frames_*.txt found"
    with open(frame_files[0]) as fp:
        frame_paths = [l.strip() for l in fp if l.strip()]
    T_total = len(frame_paths)

    query_files = sorted(f for f in os.listdir(".") if f.startswith("query_") and f.endswith(".txt"))
    assert query_files, "No query_*.txt found"
    queries = []
    for qf in query_files:
        oid = qf[len("query_"):-len(".txt")]
        with open(qf) as fp:
            lines = [l.strip() for l in fp if l.strip()]
        offset = int(lines[0])
        state = parse_region(lines[1])
        queries.append((oid, offset, state))

    all_imgs = load_frames(frame_paths)
    resized, H, W, sx, sy = resize_frames(all_imgs, MAX_SIZE)
    print(f"Frames: {T_total}, input size: {H}x{W}, CoTracker internal: 384x512 (quality independent of input size)")

    video = to_tensor(resized)  # (1, T, 3, H, W) in [0, 255]

    model = load_model()

    # Build query tensor: (1, N, 3) — each row is (t, x, y) in resized pixel space
    ct_queries = []
    for oid, offset, state in queries:
        if isinstance(state, Point):
            qx = state.x * sx
            qy = state.y * sy
        else:
            qx, qy = W / 2.0, H / 2.0
        ct_queries.append([float(offset), qx, qy])

    ct_queries = torch.tensor(ct_queries, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)

    print(f"Tracking {len(queries)} queries over {T_total} frames...")
    with torch.no_grad():
        pred_tracks, pred_visibility = model(video, queries=ct_queries)
    # pred_tracks: (1, T, N, 2) in resized pixel space

    for i, (oid, offset, state) in enumerate(queries):
        with open(f"output_{oid}.txt", "w") as fp:
            for t in range(T_total):
                if t < offset:
                    fp.write("0\n")
                else:
                    tx = float(pred_tracks[0, t, i, 0]) / sx
                    ty = float(pred_tracks[0, t, i, 1]) / sy
                    fp.write(f"{tx},{ty}\n")

    print("Done.")


if __name__ == "__main__":
    main()
