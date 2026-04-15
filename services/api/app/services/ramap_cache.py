#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/services/ramap_cache.py
# @IDE     : vscode

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np

RAMAP_DIR_NAME = "RAMaps"
RAMAP_STATUS_FILENAME = ".ramap_cache_status.json"
RAMAP_MANIFEST_FILENAME = ".ramap_cache_manifest.tsv"
RAMAP_LOG_FILENAME = ".ramap_cache.log"


def ramap_path_for_cube(cube_path: Path, ramap_root: Path) -> Path:
    return ramap_root / f"{cube_path.stem}.npy"


def is_ready_ramap(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 128


def convert_cube_to_ramap(cube_path: Path, ramap_path: Path) -> str:
    if is_ready_ramap(ramap_path):
        return str(ramap_path)

    import scipy.io

    data = scipy.io.loadmat(cube_path, variable_names=["radarCube"])
    cube = data.get("radarCube")
    if cube is None:
        raise ValueError(f"Radar cube variable 'radarCube' not found in {cube_path}")

    cube_array = np.asarray(cube, dtype=np.float32)
    if cube_array.ndim == 3:
        # RaDelft raw cubes are range x doppler x azimuth.
        ramap = np.max(cube_array, axis=1)
    elif cube_array.ndim == 2:
        ramap = cube_array
    else:
        raise ValueError(f"Unsupported radar cube shape {cube_array.shape} in {cube_path}")

    ramap = np.nan_to_num(ramap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    ramap_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ramap_path.with_name(f".{ramap_path.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.save(handle, ramap)
        os.replace(temp_path, ramap_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return str(ramap_path)


def write_status(status_path: Path, status: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(temp_path, status_path)


def _worker_count(total_items: int) -> int:
    configured = os.getenv("RADELFT_RAMAP_PREPROCESS_WORKERS")
    if configured:
        try:
            return max(1, min(int(configured), total_items))
        except ValueError:
            pass
    return max(1, min(6, total_items))


class RamapCacheManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[str, subprocess.Popen] = {}

    def ensure_cache(self, *, sequence_name: str, radar_files: Iterable[Path], ramap_root: Path) -> dict:
        radar_file_list = list(radar_files)
        total = len(radar_file_list)
        existing_status = self.get_status(ramap_root)
        if (
            existing_status
            and int(existing_status.get("total_frames") or -1) == total
            and existing_status.get("intensity_available")
            and not existing_status.get("running")
        ):
            return existing_status

        pairs = [(cube_path, ramap_path_for_cube(cube_path, ramap_root)) for cube_path in radar_file_list]
        ready_count = sum(1 for _, ramap_path in pairs if is_ready_ramap(ramap_path))
        missing_pairs = [(cube_path, ramap_path) for cube_path, ramap_path in pairs if not is_ready_ramap(ramap_path)]
        workers = _worker_count(len(missing_pairs)) if missing_pairs else 0
        status_path = ramap_root / RAMAP_STATUS_FILENAME

        if not missing_pairs:
            status = self._status_payload(
                sequence_name=sequence_name,
                ramap_root=ramap_root,
                total_frames=total,
                ready_frames=ready_count,
                missing_frames=0,
                running=False,
                workers=0,
            )
            self._safe_write_status(status_path, status)
            return status

        with self._lock:
            existing = self._processes.get(str(ramap_root))
            if existing and existing.poll() is None:
                return self.get_status(ramap_root) or self._status_payload(
                    sequence_name=sequence_name,
                    ramap_root=ramap_root,
                    total_frames=total,
                    ready_frames=ready_count,
                    missing_frames=len(missing_pairs),
                    running=True,
                    workers=workers,
                )

            status = self._status_payload(
                sequence_name=sequence_name,
                ramap_root=ramap_root,
                total_frames=total,
                ready_frames=ready_count,
                missing_frames=len(missing_pairs),
                running=True,
                workers=workers,
            )
            self._safe_write_status(status_path, status)
            self._write_manifest(ramap_root / RAMAP_MANIFEST_FILENAME, missing_pairs)
            process = self._start_precompute_process(
                sequence_name=sequence_name,
                ramap_root=ramap_root,
                workers=workers,
            )
            self._processes[str(ramap_root)] = process
            return status

    def get_status(self, ramap_root: Path) -> dict | None:
        status_path = ramap_root / RAMAP_STATUS_FILENAME
        if not status_path.exists():
            return None
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        with self._lock:
            process = self._processes.get(str(ramap_root))
            if process and process.poll() is None:
                status["running"] = True
        return status

    def _start_precompute_process(self, *, sequence_name: str, ramap_root: Path, workers: int) -> subprocess.Popen:
        api_root = Path(__file__).resolve().parents[2]
        log_path = ramap_root / RAMAP_LOG_FILENAME
        command = [
            sys.executable,
            "-m",
            "app.services.ramap_precompute_cli",
            "--sequence-name",
            sequence_name,
            "--ramap-root",
            str(ramap_root),
            "--manifest",
            str(ramap_root / RAMAP_MANIFEST_FILENAME),
            "--status",
            str(ramap_root / RAMAP_STATUS_FILENAME),
            "--workers",
            str(workers),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(api_root)
        ramap_root.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            return subprocess.Popen(
                command,
                cwd=str(api_root),
                env=env,
                stdout=log_file,
                stderr=log_file,
                close_fds=True,
            )

    def _write_manifest(self, manifest_path: Path, pairs: list[tuple[Path, Path]]) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{cube_path}\t{ramap_path}" for cube_path, ramap_path in pairs]
        manifest_path.write_text("\n".join(lines), encoding="utf-8")

    def _status_payload(
        self,
        *,
        sequence_name: str,
        ramap_root: Path,
        total_frames: int,
        ready_frames: int,
        missing_frames: int,
        running: bool,
        workers: int,
    ) -> dict:
        now = time.time()
        return {
            "sequence_name": sequence_name,
            "ramap_root": str(ramap_root),
            "total_frames": total_frames,
            "ready_frames": ready_frames,
            "missing_frames": missing_frames,
            "processed_frames": 0,
            "failed_frames": 0,
            "running": running,
            "workers": workers,
            "progress": ready_frames / total_frames if total_frames else 1.0,
            "intensity_available": total_frames > 0 and ready_frames >= total_frames and not running,
            "started_at": now if running else None,
            "updated_at": now,
            "completed_at": None if running else now,
            "error": None,
            "failures": [],
        }

    def _safe_write_status(self, status_path: Path, status: dict) -> None:
        try:
            write_status(status_path, status)
        except OSError as exc:
            status["running"] = False
            status["error"] = str(exc)


ramap_cache_manager = RamapCacheManager()
