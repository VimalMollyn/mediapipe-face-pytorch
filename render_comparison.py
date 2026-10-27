"""Render a side-by-side comparison of MediaPipe's and face_mesh_torch's outputs on one image.

Writes a figure with MediaPipe's landmarks, the PyTorch landmarks, and both overlaid, plus the
bit-exactness statistics for landmarks, blendshapes and the transformation matrix.

Usage:
    python render_comparison.py image.png docs/comparison.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import compare_with_mediapipe as cmp
import face_mesh_torch


def panel(rgb, points, color, marker, size):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for x, y in points:
        p = (int(round(x)), int(round(y)))
        if marker == "dot":
            cv2.circle(img, p, size, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (p[0] - size, p[1] - size), (p[0] + size, p[1] + size), color, 1, cv2.LINE_AA)
            cv2.line(img, (p[0] - size, p[1] + size), (p[0] + size, p[1] - size), color, 1, cv2.LINE_AA)
    return img


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    torch.set_num_threads(1)
    pixels, mp_lms, mp_bs, mp_mats = cmp.mediapipe_outputs(src, 1)
    result = face_mesh_torch.FaceLandmarkerTorch()(pixels)
    t_lms = result["face_landmarks"][0].numpy()
    t_bs = result["face_blendshapes"][0].numpy()
    t_mat = result["facial_transformation_matrixes"][0].numpy()
    h, w = pixels.shape[:2]
    rgb = pixels.numpy()

    # crop around the face with a margin, then scale panels to a fixed height
    px = mp_lms[0][:, :2] * (w, h)
    x0, y0 = np.floor(px.min(0) - 0.25 * (px.max(0) - px.min(0))).astype(int).clip(0)
    x1, y1 = np.ceil(px.max(0) + 0.25 * (px.max(0) - px.min(0))).astype(int)
    x1, y1 = min(x1, w), min(y1, h)
    crop = rgb[y0:y1, x0:x1]
    scale = 420 / crop.shape[0]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mp_pts = (mp_lms[0][:, :2] * (w, h) - (x0, y0)) * scale
    t_pts = (t_lms[:, :2] * (w, h) - (x0, y0)) * scale

    green, red = (60, 200, 60), (40, 40, 230)
    a = panel(crop, mp_pts, green, "dot", 2)
    b = panel(crop, t_pts, red, "cross", 3)
    c = panel(cv2.cvtColor(panel(crop, mp_pts, green, "dot", 2), cv2.COLOR_BGR2RGB), t_pts, red, "cross", 3)
    gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
    row = np.hstack([a, gap, b, gap, c])

    header = np.full((60, row.shape[1], 3), 255, np.uint8)
    titles = ["MediaPipe (478 landmarks)", "PyTorch port", "overlay"]
    for i, t in enumerate(titles):
        cv2.putText(header, t, (i * (a.shape[1] + 12) + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    def stat(name, x, y):
        bad = int((x != y).sum())
        return f"{name}: {'bit-exact' if bad == 0 else f'{bad}/{x.size} differ'} (max |d| = {np.abs(x - y).max():.1e})"

    lines = [stat("landmarks (478 x 3)", mp_lms[0], t_lms), stat("blendshapes (52)", mp_bs[0], t_bs),
             stat("transformation matrix (4 x 4)", mp_mats[0], t_mat)]
    footer = np.full((30 * len(lines) + 20, row.shape[1], 3), 255, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(footer, line, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([header, row, footer]))
    print("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
