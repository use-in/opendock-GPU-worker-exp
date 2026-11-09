
"""Small/Medium/Large S3 batch entrypoint; every pool uses an H100.

Requires Modal Secret 'opendock-s3'. This is NOT a Step Functions callback.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_app import MODEL_CACHE_DIR, image, model_volume
from modal_batch_routed import router_image

app = modal.App("opendock-boltz2-routed-s3")


@app.function(
    image=router_image, cpu=1, memory=1024, timeout=120,
    max_containers=1, scaledown_window=5,
)
def make_plan_s3(payload: dict, shard_size: int) -> dict:
    from worker.gpu_router import plan_batch
    return plan_batch(payload, shard_size=shard_size, allow_demo=False)


def _run_s3_shard(payload: dict, pool: str) -> dict:
    from worker.batch import execute_shard
    try:
        report = execute_shard(payload, model_cache=MODEL_CACHE_DIR)
        report["pool"] = pool
        report["gpu_profile"] = "H100"
        return report
    finally:
        model_volume.commit()


@app.function(
    image=image, gpu="H100", cpu=4, memory=16384, timeout=3600,
    max_containers=1, scaledown_window=15, retries=0,
    secrets=[modal.Secret.from_name("opendock-s3")],
    volumes={str(MODEL_CACHE_DIR.parent): model_volume},
)
def execute_small_s3(payload: dict) -> dict:
    return _run_s3_shard(payload, "small")


@app.function(
    image=image, gpu="H100", cpu=4, memory=16384, timeout=3600,
    max_containers=1, scaledown_window=15, retries=0,
    secrets=[modal.Secret.from_name("opendock-s3")],
    volumes={str(MODEL_CACHE_DIR.parent): model_volume},
)
def execute_medium_s3(payload: dict) -> dict:
    return _run_s3_shard(payload, "medium")


@app.function(
    image=image, gpu="H100", cpu=4, memory=16384, timeout=3600,
    max_containers=1, scaledown_window=15, retries=0,
    secrets=[modal.Secret.from_name("opendock-s3")],
    volumes={str(MODEL_CACHE_DIR.parent): model_volume},
)
def execute_large_s3(payload: dict) -> dict:
    return _run_s3_shard(payload, "large")


def _load(batch_file: str, max_ligands: int) -> dict:
    payload = json.loads(Path(batch_file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("ligands"), list):
        raise ValueError("Expected batch JSON with ligands list")
    if type(max_ligands) is not int or not 1 <= max_ligands <= 200:
        raise ValueError("max_ligands must be in [1,200]")
    if len(payload["ligands"]) > max_ligands:
        raise ValueError("Batch exceeds --max-ligands; increase intentionally")
    return payload


def _preview(plan: dict):
    public = {k: v for k, v in plan.items() if k != "shards"}
    public["shards"] = [
        {k: v for k, v in shard.items() if k != "payload"} for shard in plan["shards"]
    ]
    print(json.dumps(public, indent=2, ensure_ascii=False))


@app.local_entrypoint()
def preview(batch_file: str, shard_size: int = 2, max_ligands: int = 2):
    payload = _load(batch_file, max_ligands)
    plan = make_plan_s3.remote(payload, shard_size)
    _preview(plan)


@app.local_entrypoint()
def submit(
    batch_file: str,
    shard_size: int = 2,
    max_ligands: int = 2,
    max_shards: int = 1,
):
    payload = _load(batch_file, max_ligands)
    plan = make_plan_s3.remote(payload, shard_size)
    if type(max_shards) is not int or not 1 <= max_shards <= 200:
        raise ValueError("max_shards must be in [1,200]")
    if plan["shard_count"] > max_shards:
        raise ValueError("Increase --max-shards intentionally")
    _preview(plan)
    reports = []
    pool_functions = {
        "small": execute_small_s3,
        "medium": execute_medium_s3,
        "large": execute_large_s3,
    }
    for shard in plan["shards"]:
        pool = shard["pool"]
        report = pool_functions[pool].remote(shard["payload"])
        reports.append(
            {
                "part": shard["part"],
                "pool": pool,
                "gpu": "H100",
                "ligand_ids": shard["ligand_ids"],
                "max_routing_tokens": shard["max_routing_tokens"],
                "elapsed_seconds": report["elapsed_seconds"],
                "shard_manifest": (
                    payload["output_s3_prefix"]
                    + f"part-{shard['part']:03d}/shard_manifest.json"
                ),
            }
        )
    summary = {
        "batch_id": plan["batch_id"], "status": "succeeded",
        "attempt": plan["attempt"], "ligand_count": plan["ligand_count"],
        "shards": reports,
    }
    Path(f"routed-batch-summary-{plan['batch_id']}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
