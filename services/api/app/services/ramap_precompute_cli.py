#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2026-04-15
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/RadarAnnotationToolkit
# @File    : services/api/app/services/ramap_precompute_cli.py
# @IDE     : vscode

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from app.services.ramap_cache import convert_cube_to_ramap, is_ready_ramap, write_status


def _convert_pair(pair: tuple[str, str]) -> str:
    return convert_cube_to_ramap(Path(pair[0]), Path(pair[1]))


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cube_path, ramap_path = line.split("\t", 1)
        pairs.append((cube_path, ramap_path))
    return pairs


def _status(
    *,
    sequence_name: str,
    ramap_root: Path,
    total_frames: int,
    ready_frames: int,
    missing_frames: int,
    processed_frames: int,
    failed_frames: int,
    running: bool,
    workers: int,
    started_at: float,
    error: str | None = None,
    failures: list[str] | None = None,
) -> dict:
    now = time.time()
    return {
        "sequence_name": sequence_name,
        "ramap_root": str(ramap_root),
        "total_frames": total_frames,
        "ready_frames": ready_frames,
        "missing_frames": missing_frames,
        "processed_frames": processed_frames,
        "failed_frames": failed_frames,
        "running": running,
        "workers": workers,
        "progress": ready_frames / total_frames if total_frames else 1.0,
        "intensity_available": total_frames > 0 and ready_frames >= total_frames and not running and error is None,
        "started_at": started_at,
        "updated_at": now,
        "completed_at": now if not running else None,
        "error": error,
        "failures": (failures or [])[-5:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute RaDelft RAMap .npy previews from RadarCubes .mat files.")
    parser.add_argument("--sequence-name", required=True)
    parser.add_argument("--ramap-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    ramap_root = Path(args.ramap_root)
    status_path = Path(args.status)
    pairs = _read_manifest(Path(args.manifest))
    total_frames = sum(1 for path in ramap_root.glob("Pow_Frame_*.npy")) + len(pairs)
    ready_frames = sum(1 for path in ramap_root.glob("Pow_Frame_*.npy") if is_ready_ramap(path))
    missing_frames = len(pairs)
    processed_frames = 0
    failed_frames = 0
    failures: list[str] = []
    started_at = time.time()
    workers = max(1, min(args.workers, len(pairs) or 1))

    write_status(
        status_path,
        _status(
            sequence_name=args.sequence_name,
            ramap_root=ramap_root,
            total_frames=total_frames,
            ready_frames=ready_frames,
            missing_frames=missing_frames,
            processed_frames=processed_frames,
            failed_frames=failed_frames,
            running=bool(pairs),
            workers=workers,
            started_at=started_at,
            failures=failures,
        ),
    )

    if not pairs:
        return 0

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_convert_pair, pair): pair for pair in pairs}
            for future in as_completed(futures):
                cube_path, _ = futures[future]
                try:
                    future.result()
                    ready_frames += 1
                    processed_frames += 1
                except Exception as exc:
                    failed_frames += 1
                    failures.append(f"{Path(cube_path).name}: {exc}")
                missing_frames = max(0, missing_frames - 1)
                write_status(
                    status_path,
                    _status(
                        sequence_name=args.sequence_name,
                        ramap_root=ramap_root,
                        total_frames=total_frames,
                        ready_frames=ready_frames,
                        missing_frames=missing_frames,
                        processed_frames=processed_frames,
                        failed_frames=failed_frames,
                        running=True,
                        workers=workers,
                        started_at=started_at,
                        failures=failures,
                    ),
                )
    except Exception as exc:
        write_status(
            status_path,
            _status(
                sequence_name=args.sequence_name,
                ramap_root=ramap_root,
                total_frames=total_frames,
                ready_frames=ready_frames,
                missing_frames=missing_frames,
                processed_frames=processed_frames,
                failed_frames=failed_frames,
                running=False,
                workers=workers,
                started_at=started_at,
                error=str(exc),
                failures=failures,
            ),
        )
        return 1

    write_status(
        status_path,
        _status(
            sequence_name=args.sequence_name,
            ramap_root=ramap_root,
            total_frames=total_frames,
            ready_frames=ready_frames,
            missing_frames=missing_frames,
            processed_frames=processed_frames,
            failed_frames=failed_frames,
            running=False,
            workers=workers,
            started_at=started_at,
            failures=failures,
        ),
    )
    return 0 if failed_frames == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
