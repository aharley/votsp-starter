"""AllTracker wrapper for VOT folder protocol.

Reads frames_color.txt and query_*.txt from the current directory,
runs AllTracker to track each queried point, and writes output_*.txt.

Usage (via trackers.ini):
    [AllTracker]
    command = alltracker_tracker
    protocol = folderpython
    paths = /Users/aharley/votp:/Users/aharley/vot/alltracker
"""

import os
import numpy as np
import cv2
import torch
from collections import defaultdict

# Make .cuda() a no-op on CPU so the model's internal cuda() calls don't fail
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

from vot.region.io import parse_region
from vot.region import Point
from nets.alltracker import Net
from nets.blocks import InputPadder

MAX_SIZE = 1024   # max image dimension
MAX_TOKENS = 9216  # max H8*W8; bounds correlation memory to ~5.4 GB (16 * 9216^2 * 4 bytes)


def load_model():
    model = Net(16)
    url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
    print(f"Loading weights from {url} ...")
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("Model loaded.")
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
    # Also cap scale so H8*W8 stays within token budget (prevents OOM in correlation)
    token_scale = (MAX_TOKENS * 64 / (H0 * W0)) ** 0.5
    scale = min(scale, token_scale)
    H = max(8, (int(H0 * scale) // 8) * 8)
    W = max(8, (int(W0 * scale) // 8) * 8)
    resized = [cv2.resize(img, (W, H)) for img in imgs]
    return resized, H, W, W / W0, H / H0


_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def frames_to_tensor(frames):
    arr = np.stack(frames, axis=0)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # (N, 3, H, W)
    t = t / 255.0
    t = (t - _MEAN) / _STD
    return t


def track_queries(model, resized, offset, T_total, H, W, sx, sy, query_list, iters=4):
    """
    Replicate the forward_sliding loop but sample at query pixels only,
    never materializing the dense (B, T, 2, H, W) flow tensor.

    resized: full list of all frames (list of numpy HxWx3 uint8)
    query_list: [(oid, qx_i, qy_i)] in model pixel space
    Returns: {oid: [(tx, ty) in original image space]} for frames offset..T_total-1
    """
    S = model.seqlen       # sliding window length (16)
    stride = S // 2        # window advance per step (8)
    T_active = T_total - offset

    # Compute window start indices — same logic as get_T_padded_images
    indices = []
    start = 0
    while start + S < T_active:
        indices.append(start)
        start += stride
    indices.append(start)
    Tpad = indices[-1] + S - T_active
    T_padded = T_active + Tpad

    def get_frames_tensor(rel_indices):
        """Load frames by relative index, clamping to last real frame for end-padding."""
        frames = [resized[offset + min(i, T_active - 1)] for i in rel_indices]
        return frames_to_tensor(frames)  # (len, 3, H, W)

    padder = InputPadder((1, 3, H, W))

    # Anchor frame features (computed once, reused for every window)
    anchor_padded = padder.pad(get_frames_tensor([0]))[0]  # (1, 3, H_pad, W_pad)
    fmap_anchor = model.get_fmaps(anchor_padded, 1, 1, None, False)  # (1, C, H8, W8)
    device = fmap_anchor.device
    _, C, H8, W8 = fmap_anchor.shape

    traj = {oid: [] for oid, _, _ in query_list}
    full_visited = torch.zeros(T_padded, dtype=torch.bool, device=device)
    flows8 = None
    visconfs8 = None
    fmaps2 = None

    for ii, ind in enumerate(indices):
        ara = np.arange(ind, ind + S)

        if ii == 0:
            flows8 = torch.zeros((1, S, 2, H8, W8), dtype=torch.float32, device=device)
            visconfs8 = torch.zeros((1, S, 2, H8, W8), dtype=torch.float32, device=device)
            w_padded = padder.pad(get_frames_tensor(ara))[0]       # (S, 3, H_pad, W_pad)
            fmaps2 = model.get_fmaps(w_padded, 1, S, None, False).reshape(1, S, C, H8, W8)
        else:
            # Slide: keep the second half of the previous window, compute the new half
            flows8 = torch.cat([
                flows8[:, stride:stride + S // 2],
                flows8[:, stride + S // 2 - 1:stride + S // 2].repeat(1, S // 2, 1, 1, 1),
            ], dim=1)
            visconfs8 = torch.cat([
                visconfs8[:, stride:stride + S // 2],
                visconfs8[:, stride + S // 2 - 1:stride + S // 2].repeat(1, S // 2, 1, 1, 1),
            ], dim=1)
            new_rel = np.arange(ind + S // 2, ind + S)
            new_padded = padder.pad(get_frames_tensor(new_rel))[0]  # (S//2, 3, H_pad, W_pad)
            new_fmaps = model.get_fmaps(new_padded, 1, S // 2, None, False).reshape(1, S // 2, C, H8, W8)
            fmaps2 = torch.cat([fmaps2[:, stride:stride + S // 2], new_fmaps], dim=1)

        flows8_flat = flows8.reshape(S, 2, H8, W8).detach()
        visconfs8_flat = visconfs8.reshape(S, 2, H8, W8).detach()

        flow_preds, _, flows8_flat, visconfs8_flat, _ = model.forward_window(
            fmap_anchor, fmaps2, visconfs8_flat,
            iters=iters, flowfeat=None, flows8=flows8_flat, is_training=False,
        )

        # Unpad the final iteration's upsampled flow: (S, 2, H_pad, W_pad) -> (S, 2, H, W)
        flow_up = padder.unpad(flow_preds[-1])

        # Determine which frames in this window are newly visited (not already covered)
        current_visiting = torch.zeros(T_padded, dtype=torch.bool, device=device)
        current_visiting[ara] = True
        to_fill = current_visiting & (~full_visited)
        to_fill_sum = to_fill.sum().item()

        # New frames are the last to_fill_sum frames of the window
        new_flow = flow_up[-to_fill_sum:]  # (to_fill_sum, 2, H, W)

        for oid, qx_i, qy_i in query_list:
            for ti in range(to_fill_sum):
                # flow (displacement from anchor frame) + anchor pixel = tracked position in model space
                tx = (float(new_flow[ti, 0, qy_i, qx_i]) + qx_i) / sx
                ty = (float(new_flow[ti, 1, qy_i, qx_i]) + qy_i) / sy
                traj[oid].append((tx, ty))

        full_visited |= current_visiting
        flows8 = flows8_flat.reshape(1, S, 2, H8, W8)
        visconfs8 = visconfs8_flat.reshape(1, S, 2, H8, W8)

    # Discard any padded-frame entries beyond the actual sequence length
    for oid, _, _ in query_list:
        traj[oid] = traj[oid][:T_active]

    return traj


def main():
    frame_files = sorted(f for f in os.listdir(".") if f.startswith("frames_") and f.endswith(".txt"))
    assert frame_files, "No frames_*.txt found in current directory"
    with open(frame_files[0]) as fp:
        frame_paths = [l.strip() for l in fp if l.strip()]
    T_total = len(frame_paths)

    query_files = sorted(f for f in os.listdir(".") if f.startswith("query_") and f.endswith(".txt"))
    assert query_files, "No query_*.txt found in current directory"
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
    print(f"Frames: {T_total}, model size: {H}x{W} (scale {sx:.3f},{sy:.3f})")

    model = load_model()

    by_offset = defaultdict(list)
    for oid, offset, state in queries:
        by_offset[offset].append((oid, state))

    results = {}

    for offset, group in sorted(by_offset.items()):
        query_list = []
        for oid, state in group:
            if isinstance(state, Point):
                qx = state.x * sx
                qy = state.y * sy
            else:
                qx, qy = W / 2.0, H / 2.0
            qx_i = min(max(int(round(qx)), 0), W - 1)
            qy_i = min(max(int(round(qy)), 0), H - 1)
            query_list.append((oid, qx_i, qy_i))

        print(f"Tracking {len(query_list)} queries from offset={offset}, {T_total - offset} frames...")
        with torch.no_grad():
            traj = track_queries(model, resized, offset, T_total, H, W, sx, sy, query_list)

        for oid, _ in group:
            results[oid] = [None] * offset + traj[oid]

    for oid, positions in results.items():
        with open(f"output_{oid}.txt", "w") as fp:
            for pos in positions:
                if pos is None:
                    fp.write("0\n")
                else:
                    fp.write(f"{pos[0]},{pos[1]}\n")

    print(f"Done. Tracked {len(results)} objects over {T_total} frames.")


if __name__ == "__main__":
    main()
