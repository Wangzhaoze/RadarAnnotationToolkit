from __future__ import annotations

import os
from pathlib import Path

from app.services.annotation_store import AnnotationStore
from app.services.dataset_registry import DatasetRegistry
from app.services.job_manager import JobManager
from app.services.project_manager import ProjectManager

dataset_registry = DatasetRegistry()
app_state_root = Path(os.getenv("APP_STATE_ROOT", "/workspace/storage/app_state"))
annotation_root = Path(os.getenv("ANNOTATION_ROOT", str(app_state_root.parent / "annotations")))
jobs_root = Path(os.getenv("JOBS_ROOT", str(app_state_root / "jobs")))

annotation_store = AnnotationStore(root=str(annotation_root))
job_manager = JobManager(
    worker_url=os.getenv("WORKER_URL", "http://localhost:8010"),
    dataset_registry=dataset_registry,
    annotation_store=annotation_store,
    jobs_root=str(jobs_root),
)
project_manager = ProjectManager(
    dataset_registry=dataset_registry,
    annotation_store=annotation_store,
    job_manager=job_manager,
    state_root=str(app_state_root),
    inactive_annotation_root=str(annotation_root),
    inactive_jobs_root=str(jobs_root),
    host_filesystem_root=os.getenv("HOST_FILESYSTEM_ROOT", "/hostfs"),
)
