
import copy
import pytest

from worker.gpu_router import (
    HEADROOM,
    LARGE_MAX_ROUTING_TOKENS,
    classify_ligand,
    plan_batch,
)

BASE = {
    "batch_id": "router-test",
    "attempt": 1,
    "protein_sequence": "ACDEFGHIKLMNPQRSTVWY" * 5,
    "msa": {"mode": "empty"},
    "ligands": [
        {"id": "ethanol", "smiles": "CCO"},
        {"id": "acid", "smiles": "CC(=O)O"},
    ],
    "options": {"sampling_steps": 20},
}


def test_small_estimate_and_shard():
    plan = plan_batch(BASE, shard_size=2, allow_demo=True)
    assert plan["shard_count"] == 1
    assert plan["shards"][0]["pool"] == "small"
    assert plan["shards"][0]["gpu"] == "H100"
    assert plan["shards"][0]["ligand_ids"] == ["ethanol", "acid"]
    assert plan["ligands"][0]["heavy_atoms"] == 3
    assert plan["ligands"][0]["estimated_tokens"] == 103
    assert plan["ligands"][0]["routing_tokens"] == 124
    assert plan["shards"][0]["payload"]["part"] == 0


def test_medium_protein_gets_h100_medium_pool():
    big = copy.deepcopy(BASE)
    big["protein_sequence"] = "A" * 900
    plan = plan_batch(big, allow_demo=True)
    assert {entry["pool"] for entry in plan["ligands"]} == {"medium"}
    assert {entry["gpu"] for entry in plan["ligands"]} == {"H100"}


def test_large_protein_gets_h100_large_pool():
    big = copy.deepcopy(BASE)
    big["protein_sequence"] = "A" * 2500
    plan = plan_batch(big, allow_demo=True)
    assert {entry["pool"] for entry in plan["ligands"]} == {"large"}
    assert {entry["gpu"] for entry in plan["ligands"]} == {"H100"}


def test_affinity_keeps_size_pool_but_reports_unvalidated_overhead():
    job = copy.deepcopy(BASE)
    job["options"]["affinity"] = True
    entry = plan_batch(job, allow_demo=True)["ligands"][0]
    assert entry["pool"] == "small"
    assert "affinity" in entry["reason"]


def test_refuse_oversized_input_before_gpu():
    huge = copy.deepcopy(BASE)
    huge["protein_sequence"] = "A" * 4500
    with pytest.raises(ValueError, match="safety limit"):
        plan_batch(huge, allow_demo=True)


def test_bad_smiles_refused_before_gpu():
    bad = copy.deepcopy(BASE)
    bad["ligands"][0]["smiles"] = "C1INVALID"
    with pytest.raises(ValueError, match="SMILES"):
        plan_batch(bad, allow_demo=True)


def test_invalid_shard_size_rejected():
    with pytest.raises(ValueError, match="shard_size"):
        plan_batch(BASE, shard_size=0, allow_demo=True)


def test_shards_keep_ligand_ids_unique_and_part_ids():
    many = copy.deepcopy(BASE)
    many["ligands"] += [
        {"id": "x", "smiles": "CCC"},
        {"id": "y", "smiles": "CCCN"},
        {"id": "z", "smiles": "CCCC"},
    ]
    plan = plan_batch(many, shard_size=2, allow_demo=True)
    assert [shard["part"] for shard in plan["shards"]] == [0, 1, 2]
    ids = [lig for shard in plan["shards"] for lig in shard["ligand_ids"]]
    assert len(ids) == len(set(ids)) == 5


def test_shard_payload_is_valid_for_existing_worker():
    from worker.batch import parse_batch
    plan = plan_batch(BASE, allow_demo=True)
    shard = plan["shards"][0]
    raw = shard["payload"]
    assert "pool" not in raw and "gpu" not in raw
    assert len(parse_batch({k: v for k, v in raw.items() if k != "part"}, allow_demo=True).ligands) == 2


def test_cutoff_calculation_is_not_exact_boltz_tokens():
    est = classify_ligand("A" * 100, "lig", "CCO")
    assert est.routing_tokens > est.estimated_tokens
    assert HEADROOM == 1.2
    assert LARGE_MAX_ROUTING_TOKENS == 5120


def test_mixed_small_medium_split_into_distinct_h100_pools():
    mixed = copy.deepcopy(BASE)
    mixed["protein_sequence"] = "A" * 840
    mixed["ligands"] = [
        {"id": "tiny", "smiles": "CCO"},
        {"id": "larger", "smiles": "C" * 40},
        {"id": "tiny2", "smiles": "CCN"},
    ]
    plan = plan_batch(mixed, shard_size=2, allow_demo=True)
    assert plan["shard_count"] == 2
    assert [shard["pool"] for shard in plan["shards"]] == ["small", "medium"]
    assert {shard["gpu"] for shard in plan["shards"]} == {"H100"}
    assert plan["shards"][0]["ligand_ids"] == ["tiny", "tiny2"]
    assert plan["shards"][1]["ligand_ids"] == ["larger"]


def test_mixed_medium_large_split_into_distinct_h100_pools():
    mixed = copy.deepcopy(BASE)
    mixed["protein_sequence"] = "A" * 2125
    mixed["ligands"] = [
        {"id": "tiny", "smiles": "CCO"},
        {"id": "larger", "smiles": "C" * 25},
    ]
    plan = plan_batch(mixed, shard_size=2, allow_demo=True)
    assert [shard["pool"] for shard in plan["shards"]] == ["medium", "large"]
    assert {shard["gpu"] for shard in plan["shards"]} == {"H100"}
