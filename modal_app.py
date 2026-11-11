"""Modal GPU health check and Boltz-2 smoke test."""

from __future__ import annotations

import json
import subprocess

from pathlib import Path

import modal


APP_NAME = "opendock-boltz2-worker"

GPU = "H100"

MODEL_CACHE_DIR = Path(
    "/model-cache/boltz"
)

DEMO_DIR = Path(
    "/demo-results"
)


app = modal.App(APP_NAME)


image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .apt_install(
        "git",
        "libgomp1",
        "libgl1"
    )
    .pip_install(
        "boltz[cuda]==2.2.1",
        "boto3>=1.34,<2",
        "PyYAML>=6,<7"
    )
    .add_local_python_source(
        "worker"
    )
)


model_volume = modal.Volume.from_name(
    "opendock-boltz-model-cache",
    create_if_missing=True
)

demo_volume = modal.Volume.from_name(
    "opendock-boltz-demo-results",
    create_if_missing=True
)


@app.function(
    image=image,
    gpu=GPU,
    timeout=180,
    max_containers=1
)
def gpu_check() -> dict:

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available in Modal container"
        )

    details = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader"
        ],
        capture_output=True,
        text=True,
        check=True
    )

    return {
        "cuda": True,
        "torch_version": torch.__version__,
        "gpu": details.stdout.strip()
    }


@app.function(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    timeout=1800,
    max_containers=1,
    scaledown_window=60,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume,
        str(DEMO_DIR): demo_volume
    }
)
def predict_demo() -> dict:

    from worker.pipeline import execute

    test_job = {
        "job_id": "modal-smoke-profile-01",
        ####"jon_id": "modal-smoke",
        "attempt": 1,

        "protein_sequence":
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",

        "ligand_smiles": "CCO",

        "msa": {
            "mode": "empty"
        },

        "options": {
            "recycling_steps": 1,
            "sampling_steps": 20,
            "diffusion_samples": 1
        }
    }

    try:

        return execute(
            test_job,
            model_cache=MODEL_CACHE_DIR,
            demo_directory=DEMO_DIR
        )

    finally:

        model_volume.commit()
        demo_volume.commit()


@app.local_entrypoint()
def gpu():

    print(
        json.dumps(
            gpu_check.remote(),
            indent=2
        )
    )


#######
"""
@app.local_entrypoint()
def smoke():

    result = predict_demo.remote()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        "Download: modal volume get "
        "opendock-boltz-demo-results "
        "modal-smoke ./modal-smoke-output"
    )
"""
#######
@app.local_entrypoint()
def smoke():

    result = predict_demo.remote()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        "Download: modal volume get "
        "opendock-boltz-demo-results "
        "modal-smoke-profile-01 ./modal-profile-download"
    )