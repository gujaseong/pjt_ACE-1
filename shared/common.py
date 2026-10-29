"""사진 탐지에 공통으로 쓰는 간단한 함수들."""

import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolo11n.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO 모델 파일이 없습니다: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    if model.names.get(15) != "cat" or model.names.get(16) != "dog":
        raise ValueError("개와 고양이를 탐지할 수 있는 YOLO 모델이 필요합니다.")
    return model


def image_files(source):
    source = Path(source)
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    if not source.is_dir():
        raise FileNotFoundError(f"사진 폴더가 없습니다: {source}")
    return sorted(path for path in source.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"읽을 수 없는 사진입니다: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise ValueError(f"결과 사진을 저장할 수 없습니다: {path}")
    encoded.tofile(path)


def save_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
