"""Background worker for async video crop jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from job_manager import JobManager
from video_crops import process_video_crops


@dataclass
class CropJobParams:
    conf: float
    sample_interval_sec: float
    max_samples: int
    track_expiry_samples: int
    embedding_dedup_threshold: float
    identify: bool
    dist_threshold: float
    no_embedding_dedup: bool


def run_crop_job(
    job_manager: JobManager,
    job_id: str,
    video_path: str,
    params: CropJobParams,
    *,
    yolo,
    get_embedding: Callable | None,
    identify_crops: Callable | None,
) -> None:
    try:
        job_manager.set_running(job_id)
        job = job_manager.get_job(job_id)
        if job is None:
            return

        output_dir = str(job.work_dir)

        def on_progress(progress: dict) -> None:
            job_manager.update_progress(job_id, progress)

        embedding_fn = None if params.no_embedding_dedup else get_embedding
        identify_fn = identify_crops if params.identify else None

        result = process_video_crops(
            video_path,
            yolo=yolo,
            conf=params.conf,
            sample_interval_sec=params.sample_interval_sec,
            max_samples=params.max_samples,
            track_expiry_samples=params.track_expiry_samples,
            output_dir=output_dir,
            get_embedding=embedding_fn,
            embedding_dedup_threshold=params.embedding_dedup_threshold,
            identify=params.identify,
            identify_crops=identify_fn,
            dist_threshold=params.dist_threshold,
            on_progress=on_progress,
        )

        zip_path = Path(result["zip_path"]) if result.get("zip_path") else None
        job_manager.complete_job(
            job_id,
            zip_path=zip_path,
            manifest=result.get("manifest", []),
            result=result,
        )
    except Exception as exc:
        job_manager.fail_job(job_id, str(exc))
    finally:
        if os.path.exists(video_path):
            os.unlink(video_path)
