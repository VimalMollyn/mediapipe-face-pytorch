"""Detect faces in an image and extract the 478-point face mesh with MediaPipe Face Landmarker.

Usage:
    python face_mesh.py path/to/image.jpg
    python face_mesh.py path/to/image.jpg --out-image annotated.png --out-json mesh.json --num-faces 2
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections
VisionRunningMode = mp.tasks.vision.RunningMode


def ensure_model(model_path: Path) -> Path:
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model to {model_path} ...", file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def detect_face_mesh(image_path: Path, model_path: Path, num_faces: int = 1):
    """Run the face landmarker on an image. Returns (mp.Image, FaceLandmarkerResult)."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=num_faces,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    mp_image = mp.Image.create_from_file(str(image_path))
    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
    return mp_image, result


def mesh_to_array(face_landmarks, width: int, height: int) -> np.ndarray:
    """Convert one face's normalized landmarks to an (N, 3) array.

    x, y are in pixels; z is relative depth scaled like x (smaller = closer to camera).
    """
    return np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in face_landmarks],
        dtype=np.float32,
    )


def draw_face_mesh(rgb_image: np.ndarray, result) -> np.ndarray:
    annotated = rgb_image.copy()
    for face_landmarks in result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
            + FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    return annotated


def result_to_dict(result, width: int, height: int) -> dict:
    faces = []
    for i, face_landmarks in enumerate(result.face_landmarks):
        face = {
            "landmarks_normalized": [[lm.x, lm.y, lm.z] for lm in face_landmarks],
            "landmarks_pixels": mesh_to_array(face_landmarks, width, height).tolist(),
        }
        if result.face_blendshapes:
            face["blendshapes"] = {
                c.category_name: c.score for c in result.face_blendshapes[i]
            }
        if result.facial_transformation_matrixes:
            face["transformation_matrix"] = np.asarray(
                result.facial_transformation_matrixes[i]
            ).tolist()
        faces.append(face)
    return {"image_width": width, "image_height": height, "faces": faces}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--num-faces", type=int, default=1)
    parser.add_argument("--out-image", type=Path, help="save annotated image here")
    parser.add_argument("--out-json", type=Path, help="save landmarks/blendshapes as JSON")
    parser.add_argument("--out-npy", type=Path, help="save (faces, 478, 3) pixel-space mesh as .npy")
    args = parser.parse_args()

    if not args.image.exists():
        parser.error(f"image not found: {args.image}")

    mp_image, result = detect_face_mesh(args.image, ensure_model(args.model), args.num_faces)
    height, width = mp_image.height, mp_image.width

    if not result.face_landmarks:
        print("No face detected.", file=sys.stderr)
        sys.exit(1)

    meshes = np.stack([mesh_to_array(f, width, height) for f in result.face_landmarks])
    print(f"Detected {len(meshes)} face(s); mesh array shape {meshes.shape} (faces, landmarks, xyz)")
    for i, mesh in enumerate(meshes):
        x0, y0 = mesh[:, :2].min(axis=0)
        x1, y1 = mesh[:, :2].max(axis=0)
        print(f"  face {i}: bbox=({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}) nose_tip=({mesh[1][0]:.1f}, {mesh[1][1]:.1f}, {mesh[1][2]:.1f})")

    if args.out_image:
        annotated = draw_face_mesh(mp_image.numpy_view()[:, :, :3], result)
        cv2.imwrite(str(args.out_image), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"Wrote {args.out_image}")
    if args.out_json:
        args.out_json.write_text(json.dumps(result_to_dict(result, width, height), indent=2))
        print(f"Wrote {args.out_json}")
    if args.out_npy:
        np.save(args.out_npy, meshes)
        print(f"Wrote {args.out_npy}")


if __name__ == "__main__":
    main()
