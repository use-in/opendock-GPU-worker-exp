"""Core Boltz-2 prediction pipeline."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import json
import time


from worker.boltz_runner import run_boltz

from worker.contracts import parse_job

from worker.preprocessing import (
    create_yaml,
    validate_msa
)

from worker.result_manager import (
    make_manifest,
    publish_results
)

from worker.storage import (
    download_msa,
    parse_s3_uri
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
            f"jobs/{job.job_id}/"
            f"attempt-{job.attempt}/"
        )

        if output_key != expected:
            raise ValueError(
                f"output_s3_prefix key must be {expected}"
            )

    start = time.monotonic()

    with TemporaryDirectory(
        prefix=f"opendock-{job.job_id}-"
    ) as working:

        root = Path(working)

        msa = None

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

        results_dir = run_boltz(
            yaml_path,
            root / "output",
            model_cache,
            job.options
        )

        manifest = make_manifest(
            job.job_id,
            job.attempt,
            results_dir
        )

        manifest["elapsed_seconds"] = round(
            time.monotonic() - start,
            2
        )

        if demo:

            import shutil

            target = (
                demo_directory / job.job_id
            )

            if target.exists():
                shutil.rmtree(target)

            shutil.copytree(
                results_dir,
                target / "output"
            )

            (
                target / "manifest.json"
            ).write_text(
                json.dumps(
                    manifest,
                    indent=2
                ),
                encoding="utf-8"
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

        return manifest