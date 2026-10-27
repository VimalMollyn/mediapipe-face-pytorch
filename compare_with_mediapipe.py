"""Check that face_mesh_torch (PyTorch only) reproduces MediaPipe's FaceLandmarker outputs.

Runs both pipelines on the same images and reports the largest absolute differences in
landmarks, blendshapes and transformation matrices. This is the only file that imports
both libraries; face_mesh_torch.py itself never imports mediapipe.

Usage:
    python compare_with_mediapipe.py image1.png [image2.jpg ...] [--num-faces N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import face_mesh
import face_mesh_torch


def mediapipe_outputs(path: Path, num_faces: int):
    mp_image, result = face_mesh.detect_face_mesh(path, face_mesh.ensure_model(face_mesh.DEFAULT_MODEL_PATH), num_faces)
    landmarks = [np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float32) for face in result.face_landmarks]
    blendshapes = [np.array([c.score for c in face], dtype=np.float32) for face in result.face_blendshapes or []]
    matrices = [np.asarray(m, dtype=np.float32) for m in result.facial_transformation_matrixes or []]
    pixels = torch.from_numpy(mp_image.numpy_view()[:, :, :3].copy())
    return pixels, landmarks, blendshapes, matrices


def max_abs_diff(a_list, b_list):
    if len(a_list) != len(b_list):
        return float("nan")
    if not a_list:
        return 0.0
    return max(float(np.abs(a - b).max()) for a, b in zip(a_list, b_list))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--num-faces", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(1)  # the exact kernels are thousands of tiny ops; thread pools only add latency
    model = face_mesh_torch.FaceLandmarkerTorch(num_faces=args.num_faces)
    ok = True
    for path in args.images:
        mp_pixels, mp_lms, mp_bs, mp_mats = mediapipe_outputs(path, args.num_faces)
        torch_pixels = face_mesh_torch.load_image(path)
        decode_same = torch.equal(mp_pixels, torch_pixels)
        # Feed the exact pixels MediaPipe decoded so decoder differences (e.g. JPEG) don't leak in.
        result = model(mp_pixels)
        t_lms = [t.numpy() for t in result["face_landmarks"]]
        t_bs = [t.numpy() for t in result["face_blendshapes"]]
        t_mats = [t.numpy() for t in result["facial_transformation_matrixes"]]

        def report(name, a, b):
            if len(a) != len(b):
                return f"{name} COUNT MISMATCH"
            if not a:
                return f"{name} n/a"
            bad = sum(int((x != y).sum()) for x, y in zip(a, b))
            total = sum(x.size for x in a)
            return f"{name} {'EXACT' if bad == 0 else f'{bad}/{total} differ, max|d|={max_abs_diff(a, b):.2e}'}"

        same_count = len(mp_lms) == len(t_lms)
        exact = same_count and all(np.array_equal(x, y) for x, y in zip(mp_lms + mp_bs + mp_mats, t_lms + t_bs + t_mats))
        ok &= exact
        print(f"{path.name}: faces mediapipe={len(mp_lms)} torch={len(t_lms)} | {report('landmarks', mp_lms, t_lms)} | "
              f"{report('blendshapes', mp_bs, t_bs)} | {report('matrix', mp_mats, t_mats)} | "
              f"torchvision decode identical={decode_same}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
