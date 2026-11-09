"""Production Modal Boltz-2 GPU Worker."""

from __future__ import annotations

import json

from pathlib import Path

import modal

from modal_app import (
    GPU,
    MODEL_CACHE_DIR,
    image,
    model_volume
)


app = modal.App(
    "opendock-boltz2-worker-prod"
)


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    timeout=1800,
    max_containers=2,
    scaledown_window=60,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume
    },
    secrets=[
        modal.Secret.from_name(
            "opendock-s3"
        )
    ],
    retries=0
)
def predict_s3(
    payload: dict
) -> dict:

    from worker.pipeline import execute

    try:

        return execute(
            payload,
            model_cache=MODEL_CACHE_DIR
        )

    finally:

        model_volume.commit()


@app.local_entrypoint()
def submit(
    job_file: str
):

    payload = json.loads(
        Path(job_file).read_text(
            encoding="utf-8"
        )
    )

    print(
        json.dumps(
            predict_s3.remote(payload),
            indent=2,
            ensure_ascii=False
        )
    )