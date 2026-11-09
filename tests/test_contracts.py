import pytest

from worker.contracts import parse_job


JOB = {
    "job_id": "test_1",
    "protein_sequence": "ACDEFGHIKLMNPQRSTVWY",
    "ligand_smiles": "CCO",
    "msa": {
        "mode": "s3",
        "s3_uri": "s3://test/msa/q.a3m"
    },
    "output_s3_prefix": "s3://test/jobs/test_1/attempt-1/"
}


def test_valid_job():

    result = parse_job(JOB)

    assert result.msa_mode == "s3"
    assert result.options.sampling_steps == 200


def test_empty_msa_is_restricted():

    with pytest.raises(
        ValueError,
        match="disabled"
    ):

        parse_job({
            **JOB,
            "msa": {
                "mode": "empty"
            }
        })

    assert parse_job(
        {
            **JOB,
            "msa": {
                "mode": "empty"
            },
            "output_s3_prefix": None
        },
        allow_demo=True
    )


def test_no_unknown_options():

    with pytest.raises(
        ValueError,
        match="Unexpected"
    ):

        parse_job({
            **JOB,
            "options": {
                "shell": "hello"
            }
        })


def test_no_cli_injection_in_id():

    with pytest.raises(ValueError):

        parse_job({
            **JOB,
            "job_id": "../escape"
        })


def test_reject_nonbool_integer():

    with pytest.raises(ValueError):

        parse_job({
            **JOB,
            "options": {
                "sampling_steps": True
            }
        })


def test_no_msa_for_production():

    with pytest.raises(ValueError):

        parse_job({
            **JOB,
            "msa": None
        })