
"""Boltz-2 pipeline with stage timings."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

from worker.boltz_runner import run_boltz
from worker.contracts import parse_job
from worker.preprocessing import create_yaml, validate_msa
from worker.result_manager import make_manifest, publish_results
from worker.storage import download_msa, parse_s3_uri


def _record(
    timings: dict[str, float],
    name: str,
    started: float
) -> None:

    seconds = round(
        time.perf_counter() - started,
        2
    )

    timings[name] = seconds

    print(
        f"[PROFILE] {name}: {seconds:.2f}s",
        flush=True
    )


def execute(
    job_payload: dict,
    *,
    model_cache: Path,
    demo_directory: Path | None = None
) -> dict:

    demo = demo_directory is not None

    job = parse_job(
        job_payload,
        allow_demo=demo
    )

    if demo and job.output_s3_prefix is not None:
        raise ValueError(
            "Demo cannot use S3 output"
        )

    if not demo:

        _, output_key = parse_s3_uri(
            job.output_s3_prefix,
            is_prefix=True
        )

        expected = (
            f"jobs/{job.job_id}/attempt-{job.attempt}/"
        )

        if output_key != expected:
            raise ValueError(
                f"output_s3_prefix key must be {expected}"
            )

    timings: dict[str, float] = {}

    start = time.perf_counter()

    print(
        f"[PROFILE] Pipeline start: "
        f"{datetime.now(timezone.utc).isoformat()}",
        flush=True
    )

    with TemporaryDirectory(
        prefix=f"opendock-{job.job_id}-"
    ) as working:

        root = Path(working)
        msa = None

        # Step 1: Input preparation
        t = time.perf_counter()

        if job.msa_mode == "s3":

            msa = root / "input.a3m"

            download_msa(
                job.msa_s3_uri,
                msa
            )

            validate_msa(
                msa,
                job.protein_sequence
            )

        from rdkit import Chem

        if Chem.MolFromSmiles(
            job.ligand_smiles
        ) is None:
            raise ValueError(
                "Invalid ligand SMILES"
            )

        yaml_path = create_yaml(
            job,
            root / "complex.yaml",
            msa
        )

        _record(
            timings,
            "input_validation_yaml",
            t
        )

        # Step 2: Boltz CLI
        t = time.perf_counter()

        try:
            results_dir = run_boltz(
                yaml_path,
                root / "output",
                model_cache,
                job.options
            )

        except Exception:

            if demo:

                target = demo_directory / job.job_id

                target.mkdir(
                    parents=True,
                    exist_ok=True
                )

                for name in (
                    "boltz.log",
                    "gpu_usage.csv"
                ):

                    source = root / name

                    if source.exists():
                        shutil.copy2(
                            source,
                            target / name
                        )

            raise

        _record(
            timings,
            "boltz_cli",
            t
        )

        # Step 3: Result collection
        t = time.perf_counter()

        manifest = make_manifest(
            job.job_id,
            job.attempt,
            results_dir
        )

        _record(
            timings,
            "results_scan_hash",
            t
        )

        manifest["elapsed_seconds"] = round(
            time.perf_counter() - start,
            2
        )

        # Step 4: Persist results
        t = time.perf_counter()

        if demo:

            target = demo_directory / job.job_id

            if target.exists():
                shutil.rmtree(target)

            shutil.copytree(
                results_dir,
                target / "output"
            )

            for name in (
                "boltz.log",
                "gpu_usage.csv"
            ):

                source = root / name

                if source.exists():
                    shutil.copy2(
                        source,
                        target / name
                    )

            manifest["demo_volume_path"] = str(
                target
            )

        else:

            manifest = publish_results(
                manifest,
                results_dir,
                job.output_s3_prefix
            )

        _record(
            timings,
            "results_persist",
            t
        )

        manifest["timings_seconds"] = timings

        manifest["total_pipeline_seconds"] = round(
            time.perf_counter() - start,
            2
        )

        if demo:

            (target / "manifest.json").write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    ensure_ascii=False
                ),
                encoding="utf-8"
            )

        print(
            "[PROFILE] Summary: "
            + json.dumps(
                timings,
                ensure_ascii=False
            ),
            flush=True
        )

        return manifest
