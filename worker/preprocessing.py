"""Build a Boltz-2 YAML from validated inputs."""

from __future__ import annotations

from pathlib import Path
import yaml

from worker.contracts import Job


def create_yaml(
    job: Job,
    yaml_path: Path,
    msa_path: Path | None
) -> Path:

    if job.msa_mode == "s3" and msa_path is None:
        raise ValueError(
            "External MSA path missing"
        )

    protein = {
        "id": "A",
        "sequence": job.protein_sequence,
        "msa": str(msa_path) if msa_path else "empty"
    }

    ligand = {
        "id": "B",
        "smiles": job.ligand_smiles
    }

    data = {
        "version": 1,
        "sequences": [
            {"protein": protein},
            {"ligand": ligand}
        ]
    }

    if job.options.affinity:
        data["properties"] = [
            {
                "affinity": {
                    "binder": "B"
                }
            }
        ]

    yaml_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    yaml_path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True
        ),
        encoding="utf-8"
    )

    return yaml_path


def validate_msa(
    path: Path,
    expected_sequence: str
) -> None:

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(
            "MSA file is missing or empty"
        )

    with path.open("r", encoding="utf-8") as f:

        if not f.readline().startswith(">"):
            raise ValueError(
                "A3M must start with a FASTA header"
            )

        query_rows: list[str] = []

        for line in f:

            if line.startswith(">"):
                break

            query_rows.append(line.strip())

    query = "".join(query_rows)

    query = "".join(
        c for c in query
        if c.isupper() and c not in "-."
    )

    if query != expected_sequence:
        raise ValueError(
            "MSA first query does not match protein_sequence"
        )