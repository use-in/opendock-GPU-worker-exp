import json

from pathlib import Path

import pytest

from worker import pipeline


PROTEIN = "ACDEFGHIKLMNPQRSTVWY"


def test_execute_demo_with_fake_boltz(
    tmp_path,
    monkeypatch
):

    def pretend_boltz(
        yaml_path,
        output_dir,
        cache_dir,
        options
    ):

        assert "msa: empty" in yaml_path.read_text()

        out = (
            output_dir
            / "predictions"
            / "complex"
        )

        out.mkdir(
            parents=True
        )

        (
            out / "complex_model_0.cif"
        ).write_text(
            "data_example\n"
        )

        (
            out / "confidence_complex_model_0.json"
        ).write_text(
            '{"confidence_score": 0.5}'
        )

        return output_dir

    monkeypatch.setattr(
        pipeline,
        "run_boltz",
        pretend_boltz
    )

    payload = {
        "job_id": "test-demo",
        "protein_sequence": PROTEIN,
        "ligand_smiles": "CCO",
        "msa": {
            "mode": "empty"
        }
    }

    result = pipeline.execute(
        payload,
        model_cache=tmp_path / "cache",
        demo_directory=tmp_path / "demo"
    )

    assert result["status"] == "succeeded"

    assert len(result["files"]) == 2

    assert (
        tmp_path
        / "demo"
        / "test-demo"
        / "manifest.json"
    ).is_file()

    assert (
        tmp_path
        / "demo"
        / "test-demo"
        / "output"
        / "predictions"
        / "complex"
        / "complex_model_0.cif"
    ).exists()


def test_bad_smiles_detected_early(
    tmp_path
):

    with pytest.raises(
        ValueError,
        match="Invalid ligand SMILES"
    ):

        pipeline.execute(
            {
                "job_id": "test-demo",
                "protein_sequence": PROTEIN,
                "ligand_smiles": "C1INVALID",
                "msa": {
                    "mode": "empty"
                }
            },
            model_cache=tmp_path / "cache",
            demo_directory=tmp_path / "demo"
        )