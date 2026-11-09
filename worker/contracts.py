"""Strict, small Job contract."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")

JOB_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
)


@dataclass(frozen=True)
class Options:
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    affinity: bool = False


@dataclass(frozen=True)
class Job:
    job_id: str
    attempt: int
    protein_sequence: str
    ligand_smiles: str
    msa_mode: str
    msa_s3_uri: str | None
    output_s3_prefix: str | None
    options: Options


def _integer(
    value: Any,
    name: str,
    minimum: int,
    maximum: int
) -> int:

    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {maximum}]"
        )

    return value


def parse_job(
    data: dict[str, Any],
    *,
    allow_demo: bool = False
) -> Job:

    if not isinstance(data, dict):
        raise ValueError("Job must be a JSON object")

    fields = {
        "job_id",
        "attempt",
        "protein_sequence",
        "ligand_smiles",
        "msa",
        "output_s3_prefix",
        "options"
    }

    if set(data) - fields:
        raise ValueError(
            f"Unknown job fields: {sorted(set(data) - fields)}"
        )

    job_id = data.get("job_id")

    if (
        not isinstance(job_id, str)
        or not JOB_ID_PATTERN.fullmatch(job_id)
    ):
        raise ValueError(
            "job_id must be 1-64 alphanumeric, "
            "underscore or hyphen characters"
        )

    attempt = _integer(
        data.get("attempt", 1),
        "attempt",
        1,
        20
    )

    sequence = data.get("protein_sequence")

    if not isinstance(sequence, str):
        raise ValueError(
            "protein_sequence must be a string"
        )

    sequence = sequence.strip().upper()

    if (
        not 20 <= len(sequence) <= 5000
        or not set(sequence) <= AMINO_ACIDS
    ):
        raise ValueError(
            "protein_sequence must be "
            "20-5000 canonical amino acids"
        )

    smiles = data.get("ligand_smiles")

    if (
        not isinstance(smiles, str)
        or not 1 <= len(smiles) <= 1024
        or smiles != smiles.strip()
    ):
        raise ValueError(
            "ligand_smiles must be a nonempty string "
            "(max 1024 chars, no surrounding whitespace)"
        )

    if any(ord(c) < 32 for c in smiles):
        raise ValueError(
            "ligand_smiles contains control characters"
        )

    msa = data.get("msa")

    if (
        not isinstance(msa, dict)
        or set(msa) - {"mode", "s3_uri"}
    ):
        raise ValueError(
            "msa must contain mode (s3 or empty) "
            "and optional s3_uri"
        )

    mode = msa.get("mode")

    if mode not in ("s3", "empty"):
        raise ValueError(
            "msa.mode must be s3 or empty"
        )

    if mode == "empty" and not allow_demo:
        raise ValueError(
            "Single-sequence mode is disabled for production; "
            "provide external MSA"
        )

    uri = msa.get("s3_uri")

    if mode == "s3":
        if (
            not isinstance(uri, str)
            or not uri.endswith(".a3m")
        ):
            raise ValueError(
                "msa.s3_uri must reference an .a3m file"
            )

    elif uri is not None:
        raise ValueError(
            "msa.s3_uri must be omitted in empty mode"
        )

    output = data.get("output_s3_prefix")

    if output is not None:
        if (
            not isinstance(output, str)
            or not output.startswith("s3://")
            or not output.endswith("/")
        ):
            raise ValueError(
                "output_s3_prefix must be a "
                "trailing-slash s3:// prefix"
            )

    if not allow_demo and output is None:
        raise ValueError(
            "output_s3_prefix is required for production jobs"
        )

    raw_options = data.get("options", {})

    if (
        not isinstance(raw_options, dict)
        or set(raw_options) - {
            "recycling_steps",
            "sampling_steps",
            "diffusion_samples",
            "affinity"
        }
    ):
        raise ValueError("Unexpected options")

    affinity = raw_options.get("affinity", False)

    if type(affinity) is not bool:
        raise ValueError(
            "options.affinity must be boolean"
        )

    options = Options(
        recycling_steps=_integer(
            raw_options.get("recycling_steps", 3),
            "recycling_steps",
            1,
            10
        ),
        sampling_steps=_integer(
            raw_options.get("sampling_steps", 200),
            "sampling_steps",
            1,
            200
        ),
        diffusion_samples=_integer(
            raw_options.get("diffusion_samples", 1),
            "diffusion_samples",
            1,
            5
        ),
        affinity=affinity
    )

    return Job(
        job_id,
        attempt,
        sequence,
        smiles,
        mode,
        uri,
        output,
        options
    )