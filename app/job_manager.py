"""In-process job store for async video crop extraction."""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class CropJob:
    job_id: str
    status: JobStatus
    progress: dict
    error: str | None
    zip_path: Path | None
    manifest: list[dict] | None
    result: dict | None
    created_at: float
    work_dir: Path


class JobManager:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, CropJob] = {}
        self.base_dir = base_dir or Path(tempfile.gettempdir()) / "mtg_crop_jobs"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_job(self) -> CropJob:
        job_id = uuid.uuid4().hex
        work_dir = self.base_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)

        job = CropJob(
            job_id=job_id,
            status="queued",
            progress={},
            error=None,
            zip_path=None,
            manifest=None,
            result=None,
            created_at=time.time(),
            work_dir=work_dir,
        )

        with self._lock:
            self._jobs[job_id] = job

        return job

    def get_job(self, job_id: str) -> CropJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return _copy_job(job)

    def set_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "running"

    def update_progress(self, job_id: str, progress: dict) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.progress = dict(progress)

    def complete_job(
        self,
        job_id: str,
        *,
        zip_path: Path | None,
        manifest: list[dict],
        result: dict,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "completed"
            job.zip_path = zip_path
            job.manifest = manifest
            job.result = result
            job.progress = dict(result.get("video", {}))
            job.progress.update(
                {
                    "processed_samples": result.get("video", {}).get("processed_samples", 0),
                    "crops_saved": result.get("crop_count", 0),
                    "skipped_embedding_duplicates": result.get("skipped_embedding_duplicates", 0),
                }
            )

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = "failed"
            job.error = error

    def cleanup_old_jobs(self, ttl_hours: float) -> int:
        cutoff = time.time() - (ttl_hours * 3600)
        removed = 0

        with self._lock:
            stale_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.created_at < cutoff
            ]
            for job_id in stale_ids:
                job = self._jobs.pop(job_id)
                shutil.rmtree(job.work_dir, ignore_errors=True)
                removed += 1

        return removed


def _copy_job(job: CropJob) -> CropJob:
    return CropJob(
        job_id=job.job_id,
        status=job.status,
        progress=dict(job.progress),
        error=job.error,
        zip_path=job.zip_path,
        manifest=list(job.manifest) if job.manifest is not None else None,
        result=dict(job.result) if job.result is not None else None,
        created_at=job.created_at,
        work_dir=job.work_dir,
    )
