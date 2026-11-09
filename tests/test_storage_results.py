from pathlib import Path

import json
import pytest

from worker.storage import (
    parse_s3_uri,
    download_msa
)

from worker.result_manager import (
    make_manifest,
    publish_results
)


class S3Fake:

    def __init__(self):

        self.uploads = []
        self.downloads = []

    def head_object(
        self,
        *,
        Bucket,
        Key
    ):

        return {
            "ContentLength": 42
        }

    def download_file(
        self,
        bucket,
        key,
        dest
    ):

        Path(dest).write_text(
            ">q\nACDEFGHIKLMNPQRSTVWY\n"
        )

        self.downloads.append(
            (bucket, key)
        )

    def upload_file(
        self,
        filename,
        bucket,
        key,
        ExtraArgs
    ):

        self.uploads.append(
            (
                key,
                Path(filename).read_bytes()
            )
        )


def test_s3_restrictions(
    monkeypatch,
    tmp_path
):

    monkeypatch.setenv(
        "OPENDOCK_S3_BUCKET",
        "safe"
    )

    assert parse_s3_uri(
        "s3://safe/jobs/a/",
        is_prefix=True
    ) == (
        "safe",
        "jobs/a/"
    )

    for uri in [
        "https://safe/k",
        "s3://safe/../k",
        "s3://safe//k",
        "s3://safe/k?x=y",
        "s3://safe/a%2fb"
    ]:

        with pytest.raises(ValueError):
            parse_s3_uri(uri)

    with pytest.raises(
        ValueError,
        match="bucket"
    ):

        download_msa(
            "s3://other/msa/q.a3m",
            tmp_path / "msa.a3m",
            client=S3Fake()
        )

    download_msa(
        "s3://safe/msa/q.a3m",
        tmp_path / "msa.a3m",
        client=S3Fake()
    )


def test_manifest_last(
    monkeypatch,
    tmp_path
):

    monkeypatch.setenv(
        "OPENDOCK_S3_BUCKET",
        "safe"
    )

    out = tmp_path / "pred"

    (
        out / "predictions" / "complex"
    ).mkdir(
        parents=True
    )

    (
        out / "predictions"
        / "complex"
        / "complex_model_0.cif"
    ).write_text("cif")

    (
        out / "predictions"
        / "complex"
        / "confidence_complex_model_0.json"
    ).write_text(
        '{"confidence_score": 0.8}'
    )

    manifest = make_manifest(
        "job-a",
        1,
        out
    )

    fake = S3Fake()

    written = publish_results(
        manifest,
        out,
        "s3://safe/jobs/job-a/attempt-1/",
        client=fake
    )

    assert len(fake.uploads) == 3

    assert fake.uploads[-1][0].endswith(
        "manifest.json"
    )

    assert written["files"][0]["s3_uri"].startswith(
        "s3://safe/"
    )

    assert json.loads(
        fake.uploads[-1][1]
    )["status"] == "succeeded"


def test_missing_cif(tmp_path):

    tmp_path.joinpath(
        "confidence_x.json"
    ).write_text("{}")

    with pytest.raises(
        RuntimeError,
        match="no mmCIF"
    ):

        make_manifest(
            "x",
            1,
            tmp_path
        )