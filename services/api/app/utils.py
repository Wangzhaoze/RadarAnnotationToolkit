#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/utils.py
# @IDE     : vscode

from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def read_timestamps(path: Path) -> list[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def timestamp_from_path(path: Path) -> float:
    return float(path.stem)


def parse_vector_quaternion_file(path: Path) -> list[list[float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    translation = [float(value) for value in lines[0].split()]
    x, y, z, w = [float(value) for value in lines[1].split()]
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array(translation, dtype=np.float64)
    return transform.tolist()


def nearest_match(reference_ts: float, candidates: list[float], tolerance_ms: float) -> tuple[int | None, float | None]:
    if not candidates:
        return None, None
    deltas = np.abs(np.asarray(candidates, dtype=np.float64) - float(reference_ts))
    best_index = int(np.argmin(deltas))
    delta_ms = float(deltas[best_index] * 1000.0)
    if delta_ms > tolerance_ms:
        return None, delta_ms
    return best_index, delta_ms


def float_heatmap_to_png_bytes(path: Path) -> bytes:
    stat = path.stat()
    return _float_heatmap_to_png_bytes_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _float_heatmap_to_png_bytes_cached(path_str: str, mtime_ns: int, file_size: int) -> bytes:
    path = Path(path_str)
    if path.suffix.lower() == ".mat":
        import scipy.io

        payload = scipy.io.loadmat(path, variable_names=["radarCube"])
        values = payload.get("radarCube")
        if values is None:
            raise ValueError(f"Radar cube variable 'radarCube' not found in {path.name}")
    elif path.suffix.lower() == ".npy":
        values = np.load(path)
    else:
        values = np.fromfile(path, dtype=np.float32)

    image_array = np.asarray(values, dtype=np.float32)
    if image_array.ndim == 3:
        # RaDelft raw radar cubes are range x doppler x azimuth; preview them as a range-azimuth map.
        doppler_axis = 1 if path.suffix.lower() == ".mat" else 0
        image_array = np.max(image_array, axis=doppler_axis)
    elif image_array.ndim == 2 and image_array.shape[1] in {3, 4, 5, 6, 7, 8} and image_array.shape[0] > 32:
        x_values = image_array[:, 0]
        y_values = image_array[:, 1]
        weights = np.abs(image_array[:, 3]) if image_array.shape[1] > 3 else None
        bins = min(256, max(64, int(math.sqrt(image_array.shape[0]))))
        image_array, _, _ = np.histogram2d(y_values, x_values, bins=(bins, bins), weights=weights)
    elif image_array.ndim == 1:
        side = int(math.sqrt(image_array.size))
        image_array = image_array.reshape(side, side) if side * side == image_array.size else image_array.reshape(1, -1)
    elif image_array.ndim != 2:
        raise ValueError(f"Unsupported radar preview array shape: {image_array.shape}")

    image_array = np.nan_to_num(image_array, nan=0.0, posinf=0.0, neginf=0.0)
    image_array = np.log1p(np.abs(image_array))
    image_array -= image_array.min()
    upper = float(np.percentile(image_array, 99.5))
    if upper > 0:
        image_array = np.clip(image_array / upper, 0.0, 1.0)
    else:
        max_value = float(image_array.max())
        if max_value > 0:
            image_array /= max_value
    image_array = np.flipud(image_array)
    rgb_image = _apply_heatmap_colormap(image_array)
    image = Image.fromarray(rgb_image, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def radelft_radar_semantic_to_png_bytes(path: Path) -> bytes:
    stat = path.stat()
    return _radelft_radar_semantic_to_png_bytes_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=256)
def _radelft_radar_semantic_to_png_bytes_cached(path_str: str, mtime_ns: int, file_size: int) -> bytes:
    labels = np.load(Path(path_str))
    label_image = np.asarray(labels)
    if label_image.ndim == 3:
        # RaDelft semantic cubes are elevation x range x azimuth; collapse elevation for a 2D radar view.
        label_image = np.max(label_image, axis=0)
    elif label_image.ndim != 2:
        raise ValueError(f"Unsupported RaDelft radar semantic array shape: {label_image.shape}")

    label_image = np.nan_to_num(label_image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int32, copy=False)
    label_image = np.flipud(label_image)
    label_image = np.fliplr(label_image)

    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]
    palette[1] = [255, 255, 255]
    palette[2] = [30, 30, 255]
    palette[3] = [245, 150, 100]
    palette[4] = [245, 230, 100]
    clipped = np.clip(label_image, 0, 255)
    rgb_image = palette[clipped]
    image = Image.fromarray(rgb_image, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _apply_heatmap_colormap(values: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [0.0, 26, 72, 232],
            [0.35, 0, 176, 255],
            [0.68, 255, 214, 10],
            [1.0, 220, 35, 24],
        ],
        dtype=np.float32,
    )
    normalized = np.clip(values, 0.0, 1.0)
    red = np.interp(normalized, stops[:, 0], stops[:, 1])
    green = np.interp(normalized, stops[:, 0], stops[:, 2])
    blue = np.interp(normalized, stops[:, 0], stops[:, 3])
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
