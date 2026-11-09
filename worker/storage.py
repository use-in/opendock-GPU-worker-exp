"""S3 input/output operations."""

from __future__ import annotations

import os

from pathlib import Path
from urllib.parse import urlsplit


MAX_MSA_BYTES = 100 * 1024 * 1024


def parse_s3_uri(
    uri: str,
    *,
    is_prefix: bool = False
) -> tuple[str, str]:

    if (
        not isinstance(uri, str)
        or not uri.startswith("s3://")
    ):
        raise ValueError("Expected s3:// URI")

    p = urlsplit(uri)

    if (
        p.scheme != "s3"
        or not p.netloc
        or p.query
        or p.fragment
        or not p.path.startswith("/")
    ):
        raise ValueError("Invalid S3 URI")

    key = p.path[1:]

    if (
        not key
        or key.startswith("/")
        or "//" in key
        or "\\" in key
        or "%" in key
        or any(
            seg in (".", "..")
            for seg in key.split("/")
        )
    ):
        raise ValueError("Invalid S3 key")

    if is_prefix != key.endswith("/"):
        raise ValueError(
            "S3 URI prefix needs trailing slash; "
            "object URI must not have one"
        )

    return p.netloc, key


def s3_client():

    import boto3

    return boto3.client("s3")


def _allowed_bucket(bucket: str) -> None:

    configured = os.environ.get(
        "OPENDOCK_S3_BUCKET"
    )

    if not configured:
        raise RuntimeError(
            "OPENDOCK_S3_BUCKET secret is missing"
        )

    if bucket != configured:
        raise ValueError(
            "S3 URI does not match authorized bucket"
        )


def download_msa(
    uri: str,
    destination: Path,
    *,
    client=None
) -> None:

    bucket, key = parse_s3_uri(uri)

    _allowed_bucket(bucket)

    if not key.endswith(".a3m"):
        raise ValueError(
            "MSA must be .a3m"
        )

    client = client or s3_client()

    size = client.head_object(
        Bucket=bucket,
        Key=key
    )["ContentLength"]

    if not 0 < size <= MAX_MSA_BYTES:
        raise ValueError(
            "MSA exceeds max size or is empty"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    client.download_file(
        bucket,
        key,
        str(destination)
    )

    if not 0 < destination.stat().st_size <= MAX_MSA_BYTES:
        raise ValueError(
            "Downloaded MSA size invalid"
        )


def upload_file(
    local_path: Path,
    bucket: str,
    key: str,
    *,
    client=None
) -> None:

    _allowed_bucket(bucket)

    client = client or s3_client()

    suffix = local_path.suffix.lower()

    content_type = {
        ".cif": "chemical/x-mmcif",
        ".json": "application/json",
        ".npz": "application/octet-stream"
    }.get(
        suffix,
        "application/octet-stream"
    )

    client.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={
            "ContentType": content_type
        }
    )