
"""Small/Medium/Large routed Boltz-2 batch demo; ALL pools use H100.

Plan (Modal CPU only): py -m modal run modal_batch_routed.py::preview
Run (GPU charges):    py -m modal run modal_batch_routed.py::demo
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import modal

from modal_app import MODEL_CACHE_DIR, image, model_volume

APP_NAME = "opendock-boltz2-routed-demo"
RESULTS_DIR = Path("/batch-results")
app = modal.App(APP_NAME)

router_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("rdkit==2025.9.4")
    .add_local_python_source("worker")
)

results_volume = modal.Volume.from_name(
    "opendock-boltz-batch-results", create_if_missing=True
)


@app.function(
    image=router_image, cpu=1, memory=1024, timeout=120, max_containers=1,
    scaledown_window=5,
)
def make_plan(payload: dict, shard_size: int) -> dict:
    from worker.gpu_router import plan_batch
    return plan_batch(payload, shard_size=shard_size, allow_demo=True)


def _run_shard(payload: dict, pool: str) -> dict:
    from worker.batch import execute_shard

    try:
        report = execute_shard(
            payload,
            model_cache=MODEL_CACHE_DIR,
            demo_directory=RESULTS_DIR,
        )
        report["pool"] = pool
        report["gpu_profile"] = "H100"
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        report["gpu_actual"] = result.stdout.strip() if result.returncode == 0 else None
        return report
    finally:
        model_volume.commit()
        results_volume.commit()


@app.function(
    image=image, gpu="H100", cpu=4, memory=16384, timeout=3600,
    max_containers=1, scaledown_window=15, retries=0,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume,
        str(RESULTS_DIR): results_volume,
    },
)
def execute_small(payload: dict) -> dict:
    return _run_shard(payload, "small")


@app.function(
    image=image, gpu="H100", cpu=4, memory=16384, timeout=3600,
    max_containers=1, scaledown_window=15, retries=0,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume,
        str(RESULTS_DIR): results_volume,
    },
)
def execute_medium(payload: dict) -> dict:
    return _run_shard(payload, "medium")


@app.function(
    image=image, gpu="H100", cpu=4, memory=16384, timeout=3600,
    max_containers=1, scaledown_window=15, retries=0,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume,
        str(RESULTS_DIR): results_volume,
    },
)
def execute_large(payload: dict) -> dict:
    return _run_shard(payload, "large")


def _load_batch(batch_file: str, max_ligands: int) -> dict:
    payload = json.loads(Path(batch_file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("ligands"), list):
        raise ValueError("Expected batch JSON with ligands list")
    if type(max_ligands) is not int or not 1 <= max_ligands <= 200:
        raise ValueError("max_ligands must be an integer in [1,200]")
    if len(payload["ligands"]) > max_ligands:
        raise ValueError("Batch exceeds max_ligands; increase explicitly")
    return payload


def _print_plan(plan: dict) -> None:
    printable = {k: v for k, v in plan.items() if k != "shards"}
    printable["shards"] = [
        {k: v for k, v in shard.items() if k != "payload"}
        for shard in plan["shards"]
    ]
    print(json.dumps(printable, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def preview(
    batch_file: str = "examples/batch_routed_smoke.json",
    shard_size: int = 2,
    max_ligands: int = 2,
):
    payload = _load_batch(batch_file, max_ligands)
    plan = make_plan.remote(payload, shard_size)
    _print_plan(plan)


@app.local_entrypoint()
def demo(
    batch_file: str = "examples/batch_routed_smoke.json",
    shard_size: int = 2,
    max_ligands: int = 2,
    max_shards: int = 1,
):
    """Sequential shard calls cap this CLI run to one active H100 at a time."""
    payload = _load_batch(batch_file, max_ligands)
    plan = make_plan.remote(payload, shard_size)
    if type(max_shards) is not int or not 1 <= max_shards <= 200:
        raise ValueError("max_shards must be in [1,200]")
    if plan["shard_count"] > max_shards:
        raise ValueError(
            f"Plan needs {plan['shard_count']} GPU shards; "
            "increase --max-shards intentionally"
        )
    _print_plan(plan)
    reports = []
    pool_functions = {
        "small": execute_small,
        "medium": execute_medium,
        "large": execute_large,
    }
    for shard in plan["shards"]:
        pool = shard["pool"]
        print(
            f"Starting part-{shard['part']:03d}, pool={pool}, gpu=H100, "
            f"ligands={','.join(shard['ligand_ids'])}", flush=True,
        )
        report = pool_functions[pool].remote(shard["payload"])
        reports.append(
            {
                "part": report["part"],
                "pool": pool,
                "gpu": "H100",
                "max_routing_tokens": shard["max_routing_tokens"],
                "ligand_ids": shard["ligand_ids"],
                "elapsed_seconds": report["elapsed_seconds"],
                "gpu_actual": report["gpu_actual"],
            }
        )
        print(f"Completed part-{shard['part']:03d}", flush=True)

    summary = {
        "batch_id": plan["batch_id"],
        "attempt": plan["attempt"],
        "status": "succeeded",
        "ligand_count": plan["ligand_count"],
        "shard_count": len(reports),
        "results_volume": "opendock-boltz-batch-results",
        "shards": reports,
    }
    Path(f"routed-batch-summary-{plan['batch_id']}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Download: py -m modal volume ls opendock-boltz-batch-results")
