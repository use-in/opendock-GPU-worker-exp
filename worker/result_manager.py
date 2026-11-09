"""Discover and upload Boltz-2 prediction artifacts."""

from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any

from worker.storage import (
    parse_s3_uri,
    upload_file
)


EXTENSIONS = {
    ".cif",
    ".json",
    ".npz"
}

MAX_RESULT_BYTES = 500 * 1024 * 1024


def discover_files(
    output_dir: Path
) -> list[Path]:

    artifacts = sorted(
        p for p in output_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    )

    if not any(
        p.suffix.lower() == ".cif"
        for p in artifacts
    ):
        raise RuntimeError(
            "Boltz returned no mmCIF structure files"
        )

    if not any(
        p.name.startswith("confidence_")
        and p.suffix == ".json"
        for p in artifacts
    ):
        raise RuntimeError(
            "Boltz returned no confidence JSON"
        )

    return artifacts


def make_manifest(
    job_id: str,
    attempt: int,
    output_dir: Path
) -> dict[str, Any]:

    artifacts = discover_files(output_dir)

    files = []

    for p in artifacts:

        if p.stat().st_size > MAX_RESULT_BYTES:
            raise RuntimeError(
                f"Output artifact exceeds maximum size: {p.name}"
            )

        with p.open("rb") as handle:
            digest = hashlib.file_digest(
                handle,
                "sha256"
            ).hexdigest()

        files.append({
            "path": p.relative_to(
                output_dir
            ).as_posix(),

            "size_bytes": p.stat().st_size,

            "sha256": digest
        })

    return {
        "schema_version": 1,
        "job_id": job_id,
        "attempt": attempt,
        "model": "boltz2",
        "status": "succeeded",
        "files": files
    }


def publish_results(
    manifest: dict[str, Any],
    output_dir: Path,
    uri: str,
    *,
    client=None
) -> dict[str, Any]:

    bucket, prefix = parse_s3_uri(
        uri,
        is_prefix=True
    )

    for item in manifest["files"]:

        artifact = output_dir / item["path"]

        key = (
            prefix
            + "artifacts/"
            + item["path"]
        )

        upload_file(
            artifact,
            bucket,
            key,
            client=client
        )

        item["s3_uri"] = (
            f"s3://{bucket}/{key}"
        )

    manifest_path = (
        output_dir.parent / "manifest.json"
    )

    manifest["manifest_s3_uri"] = (
        f"s3://{bucket}/{prefix}manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    upload_file(
        manifest_path,
        bucket,
        prefix + "manifest.json",
        client=client
    )

    return manifest