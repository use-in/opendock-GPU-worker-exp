from pathlib import Path

import yaml
import pytest

from worker.contracts import parse_job

from worker.preprocessing import (
    create_yaml,
    validate_msa
)

from worker.boltz_runner import make_command


JOB = {
    "job_id": "test_1",
    "protein_sequence": "ACDEFGHIKLMNPQRSTVWY",
    "ligand_smiles": "CCO",

    "msa": {
        "mode": "s3",
        "s3_uri": "s3://test/msa/q.a3m"
    },

    "output_s3_prefix": "s3://test/jobs/test_1/attempt-1/",

    "options": {
        "affinity": True
    }
}


def test_yaml_and_cli(tmp_path):

    job = parse_job(JOB)

    msa = tmp_path / "query.a3m"

    msa.write_text(
        ">q\n"
        "ACDEFGHIKLMNPQRSTVWY\n"
        ">s\n"
        "ACDEFGHIKLMNPQRSTVWY\n"
    )

    validate_msa(
        msa,
        job.protein_sequence
    )

    path = create_yaml(
        job,
        tmp_path / "input.yaml",
        msa
    )

    parsed = yaml.safe_load(
        path.read_text()
    )

    assert (
        parsed["sequences"][0]["protein"]["msa"]
        == str(msa)
    )

    assert (
        parsed["sequences"][1]["ligand"]["smiles"]
        == "CCO"
    )

    assert parsed["properties"] == [
        {
            "affinity": {
                "binder": "B"
            }
        }
    ]

    cmd = make_command(
        path,
        tmp_path / "result",
        tmp_path / "cache",
        job.options
    )

    assert cmd[:3] == [
        "boltz",
        "predict",
        str(path)
    ]

    assert "--use_msa_server" not in cmd

    assert cmd[
        cmd.index("--model") + 1
    ] == "boltz2"


def test_mismatch_query(tmp_path):

    p = tmp_path / "x.a3m"

    p.write_text(
        ">query\n"
        "AAAAAAAAAAAAAAAAAAAA\n"
    )

    with pytest.raises(
        ValueError,
        match="does not match"
    ):

        validate_msa(
            p,
            "ACDEFGHIKLMNPQRSTVWY"
        )