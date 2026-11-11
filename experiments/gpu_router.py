"""Route Boltz-2 jobs to three H100 execution pools."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from worker.batch import MAX_SHARD_LIGANDS, parse_batch


SMALL_MAX_ROUTING_TOKENS = 1024
MEDIUM_MAX_ROUTING_TOKENS = 2560
LARGE_MAX_ROUTING_TOKENS = 5120

HEADROOM = 1.20

GPU_TYPE = "H100"
POOL_ORDER = ("small", "medium", "large")


@dataclass(frozen=True)
class LigandEstimate:
    ligand_id: str
    heavy_atoms: int
    estimated_tokens: int
    routing_tokens: int
    pool: str
    gpu: str
    reason: str


def _ligand_heavy_atoms(smiles: str) -> int:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            "SMILES could not be parsed by RDKit"
        )

    atoms = molecule.GetNumHeavyAtoms()

    if atoms < 1:
        raise ValueError(
            "Ligand must have at least one heavy atom"
        )

    return atoms


def classify_ligand(
    protein_sequence: str,
    ligand_id: str,
    smiles: str,
    *,
    affinity: bool = False,
) -> LigandEstimate:

    heavy_atoms = _ligand_heavy_atoms(smiles)

    estimated = len(protein_sequence) + heavy_atoms
    routing = ceil(estimated * HEADROOM)

    if routing <= SMALL_MAX_ROUTING_TOKENS:
        pool = "small"

    elif routing <= MEDIUM_MAX_ROUTING_TOKENS:
        pool = "medium"

    elif routing <= LARGE_MAX_ROUTING_TOKENS:
        pool = "large"

    else:
        raise ValueError(
            f"Ligand {ligand_id}: estimated routing size "
            f"{routing} exceeds unbenchmarked safety limit "
            f"{LARGE_MAX_ROUTING_TOKENS}"
        )

    reason = "estimated input size (unbenchmarked proxy)"

    if affinity:
        reason += (
            "; affinity enabled "
            "(no validated VRAM adjustment)"
        )

    return LigandEstimate(
        ligand_id=ligand_id,
        heavy_atoms=heavy_atoms,
        estimated_tokens=estimated,
        routing_tokens=routing,
        pool=pool,
        gpu=GPU_TYPE,
        reason=reason,
    )


def plan_batch(
    payload: dict[str, Any],
    *,
    shard_size: int = 2,
    allow_demo: bool = False,
) -> dict[str, Any]:

    if (
        type(shard_size) is not int
        or not 1 <= shard_size <= MAX_SHARD_LIGANDS
    ):
        raise ValueError(
            f"shard_size must be between "
            f"1 and {MAX_SHARD_LIGANDS}"
        )

    batch = parse_batch(
        payload,
        allow_demo=allow_demo,
    )

    estimates = [
        classify_ligand(
            batch.protein_sequence,
            ligand.ligand_id,
            ligand.smiles,
            affinity=batch.options.affinity,
        )
        for ligand in batch.ligands
    ]

    smiles_by_id = {
        ligand.ligand_id: ligand.smiles
        for ligand in batch.ligands
    }

    shards = []

    for pool in POOL_ORDER:

        group = [
            entry for entry in estimates
            if entry.pool == pool
        ]

        for start in range(
            0,
            len(group),
            shard_size,
        ):

            chunk = group[start:start + shard_size]
            part = len(shards)

            shard_payload = dict(payload)
            shard_payload["part"] = part

            shard_payload["ligands"] = [
                {
                    "id": entry.ligand_id,
                    "smiles": smiles_by_id[entry.ligand_id],
                }
                for entry in chunk
            ]

            shards.append({
                "part": part,
                "pool": pool,
                "gpu": GPU_TYPE,
                "ligand_ids": [
                    entry.ligand_id
                    for entry in chunk
                ],
                "max_routing_tokens": max(
                    entry.routing_tokens
                    for entry in chunk
                ),
                "payload": shard_payload,
            })

    return {
        "schema_version": 2,
        "batch_id": batch.batch_id,
        "attempt": batch.attempt,
        "protein_residues": len(
            batch.protein_sequence
        ),
        "ligand_count": len(batch.ligands),
        "shard_count": len(shards),
        "routing_policy": {
            "estimator": (
                "protein residues + "
                "RDKit ligand heavy atoms"
            ),
            "routing_headroom_factor": HEADROOM,
            "gpu_type": GPU_TYPE,
            "small_max_routing_tokens":
                SMALL_MAX_ROUTING_TOKENS,
            "medium_max_routing_tokens":
                MEDIUM_MAX_ROUTING_TOKENS,
            "large_max_routing_tokens":
                LARGE_MAX_ROUTING_TOKENS,
            "measured_vram_thresholds": False,
        },
        "ligands": [
            asdict(entry)
            for entry in estimates
        ],
        "shards": shards,
    }