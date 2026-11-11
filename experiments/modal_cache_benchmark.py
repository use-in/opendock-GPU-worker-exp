
"""Compare repeated Boltz-2 CLI launches with a resident Python model.

Run:
    py -m modal run modal_cache_benchmark.py::benchmark --repeats 2

Requirements:
    - modal_app.py
    - worker/ package

GPU: H100
Input: Same protein + ethanol for every repetition
Batch: Not used

Experimental code. Resident model execution requires runtime validation.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
from tempfile import TemporaryDirectory

import modal

# Shared model cache path
MODEL_CACHE_DIR = Path("/model-cache/boltz")

# Use the same persistent Volume as modal_app.py
model_volume = modal.Volume.from_name(
    "opendock-boltz-model-cache",
    create_if_missing=True,
)

# Standalone image: does not depend on importing modal_app.py
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgomp1", "libgl1")
    .pip_install(
        "boltz[cuda]==2.2.1",
        "boto3>=1.34,<2",
        "PyYAML>=6,<7",
    )
    .add_local_python_source("worker")
)


app = modal.App("opendock-boltz-cache-benchmark")

RESULTS_ROOT = Path("/bench-results")

results_volume = modal.Volume.from_name(
    "opendock-boltz-benchmarks",
    create_if_missing=True,
)

PROTEIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED"
    "GRTLSDYNIQKESTLHLVLRLRGG"
)

BENCHMARK_LIGAND = ("ethanol", "CCO")


def _make_job(ligand_id: str, smiles: str):
    from worker.contracts import Job, Options

    return Job(
        job_id=f"bench_{ligand_id}",
        attempt=1,
        protein_sequence=PROTEIN,
        ligand_smiles=smiles,
        msa_mode="empty",
        msa_s3_uri=None,
        output_s3_prefix=None,
        options=Options(
            recycling_steps=1,
            sampling_steps=20,
            diffusion_samples=1,
            affinity=False,
        ),
    )


def _save_report(
    run_id: str,
    mode: str,
    report: dict,
):
    dest = RESULTS_ROOT / run_id / mode
    dest.mkdir(parents=True, exist_ok=True)

    (dest / "metrics.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    results_volume.commit()


def _cache_info():
    ckpt = MODEL_CACHE_DIR / "boltz2_conf.ckpt"

    return {
        "checkpoint_exists": ckpt.is_file(),
        "checkpoint_bytes": (
            ckpt.stat().st_size
            if ckpt.is_file()
            else None
        ),
    }


def _pred_counts(results_dir: Path):
    return {
        "cif_count": len(
            list(results_dir.rglob("*.cif"))
        ),
        "confidence_count": len(
            list(
                results_dir.rglob(
                    "confidence_*.json"
                )
            )
        ),
    }


# ============================================================
# Experiment A: CLI subprocess restart
# ============================================================

@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    scaledown_window=300,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume,
        str(RESULTS_ROOT): results_volume,
    },
)
def cli_benchmark(
    run_id: str,
    repeats: int = 2,
) -> dict:
    """Run identical inputs in separate Boltz CLI processes."""

    from worker.preprocessing import create_yaml
    from worker.boltz_runner import run_boltz

    if not 2 <= repeats <= 5:
        raise ValueError(
            "repeats must be between 2 and 5"
        )

    records = []
    started = time.perf_counter()

    try:
        with TemporaryDirectory(
            prefix="bench-cli-"
        ) as directory:

            root = Path(directory)

            name, smiles = BENCHMARK_LIGAND
            job = _make_job(name, smiles)

            yaml_file = create_yaml(
                job,
                root / "complex.yaml",
                None,
            )

            for index in range(1, repeats + 1):

                print(
                    f"[CLI] Starting prediction "
                    f"{index}/{repeats}",
                    flush=True,
                )

                t0 = time.perf_counter()

                result_dir = run_boltz(
                    yaml_file,
                    root / f"out-{index}",
                    MODEL_CACHE_DIR,
                    job.options,
                )

                seconds = round(
                    time.perf_counter() - t0,
                    3,
                )

                counts = _pred_counts(result_dir)

                if (
                    counts["cif_count"] < 1
                    or counts["confidence_count"] < 1
                ):
                    raise RuntimeError(
                        f"CLI output missing "
                        f"in repeat {index}: {counts}"
                    )

                records.append({
                    "repeat": index,
                    "ligand": name,
                    "cli_seconds": seconds,
                    **counts,
                })

                print(
                    f"[CLI] Prediction {index}: "
                    f"{seconds:.3f}s",
                    flush=True,
                )

        total_cli_seconds = round(
            sum(
                row["cli_seconds"]
                for row in records
            ),
            3,
        )

        report = {
            "mode": "cli_new_process_each_call",
            "gpu": "H100",
            "msa": "empty",
            "recycling_steps": 1,
            "sampling_steps": 20,
            "diffusion_samples": 1,
            "n": repeats,
            "same_input": True,
            "calls": records,
            "total_cli_seconds": total_cli_seconds,
            "cache": _cache_info(),
            "function_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        }

        _save_report(
            run_id,
            "cli",
            report,
        )

        return report

    finally:
        model_volume.commit()


# ============================================================
# Experiment B: Resident model / process caching
# ============================================================

@app.cls(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    scaledown_window=300,
    volumes={
        str(MODEL_CACHE_DIR.parent): model_volume,
        str(RESULTS_ROOT): results_volume,
    },
)
class ResidentBoltz:
    """Load the Python model once per Modal container.

    Tests model-object reuse across requests.

    Lightning may move the model from GPU to CPU during
    teardown, so GPU residency is measured explicitly.
    """

    @modal.enter()
    def load_once(self):

        import os
        import torch

        from boltz.main import (
            Boltz2DiffusionParams,
            BoltzSteeringParams,
            MSAModuleArgs,
            PairformerArgsV2,
            download_boltz2,
        )

        from boltz.model.models.boltz2 import Boltz2

        t0 = time.perf_counter()

        for key in (
            "CUEQ_DEFAULT_CONFIG",
            "CUEQ_DISABLE_AOT_TUNING",
        ):
            os.environ.setdefault(key, "1")

        torch.set_grad_enabled(False)

        torch.set_float32_matmul_precision(
            "highest"
        )

        MODEL_CACHE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        download_boltz2(MODEL_CACHE_DIR)

        print(
            "[RESIDENT] Loading Boltz-2 model...",
            flush=True,
        )

        self.model = Boltz2.load_from_checkpoint(
            MODEL_CACHE_DIR / "boltz2_conf.ckpt",
            strict=True,
            predict_args={
                "recycling_steps": 1,
                "sampling_steps": 20,
                "diffusion_samples": 1,
                "max_parallel_samples": None,
                "write_confidence_summary": True,
                "write_full_pae": False,
                "write_full_pde": False,
            },
            map_location="cpu",
            diffusion_process_args=asdict(
                Boltz2DiffusionParams()
            ),
            ema=False,
            use_kernels=True,
            pairformer_args=asdict(
                PairformerArgsV2()
            ),
            msa_args=asdict(
                MSAModuleArgs(
                    subsample_msa=False,
                    use_paired_feature=True,
                )
            ),
            steering_args=asdict(
                BoltzSteeringParams()
            ),
        )

        self.model.eval()
        self.model.to("cuda")

        torch.cuda.synchronize()

        self.init_seconds = round(
            time.perf_counter() - t0,
            3,
        )

        self.initial_vram_mb = round(
            torch.cuda.memory_allocated() / 2**20,
            1,
        )

        self.counter = 0
        self.container_started_at = time.time()

        self.gpu_name = torch.cuda.get_device_name(0)

        print(
            f"[RESIDENT] Model loaded in "
            f"{self.init_seconds:.3f}s",
            flush=True,
        )

        print(
            f"[RESIDENT] Initial GPU memory: "
            f"{self.initial_vram_mb:.1f} MiB",
            flush=True,
        )

        model_volume.commit()

    @modal.method()
    def predict(
        self,
        run_id: str,
        ligand_id: str,
        smiles: str,
    ) -> dict:

        import torch

        from pytorch_lightning import Trainer
        from rdkit import Chem

        from boltz.main import process_inputs
        from boltz.data.types import Manifest

        from boltz.data.module.inferencev2 import (
            Boltz2InferenceDataModule,
        )

        from boltz.data.write.writer import BoltzWriter

        from worker.preprocessing import create_yaml

        self.counter += 1

        request_start = time.perf_counter()

        print(
            f"[RESIDENT] Prediction "
            f"{self.counter} started",
            flush=True,
        )

        with TemporaryDirectory(
            prefix=f"bench-resident-{ligand_id}-"
        ) as tmp:

            root = Path(tmp)

            yaml_file = create_yaml(
                _make_job(ligand_id, smiles),
                root / "complex.yaml",
                None,
            )

            out_dir = root / "boltz_results_complex"
            out_dir.mkdir()

            Chem.SetDefaultPickleProperties(
                Chem.PropertyPickleOptions.AllProps
            )

            # --------------------------------
            # Preprocessing
            # --------------------------------

            preprocess_start = time.perf_counter()

            process_inputs(
                data=[yaml_file],
                out_dir=out_dir,
                ccd_path=MODEL_CACHE_DIR / "ccd.pkl",
                mol_dir=MODEL_CACHE_DIR / "mols",
                use_msa_server=False,
                msa_server_url=(
                    "https://api.colabfold.com"
                ),
                msa_pairing_strategy="greedy",
                msa_server_username=None,
                msa_server_password=None,
                api_key_header=None,
                api_key_value=None,
                boltz2=True,
                preprocessing_threads=2,
                max_msa_seqs=8192,
            )

            manifest = Manifest.load(
                out_dir
                / "processed"
                / "manifest.json"
            )

            if len(manifest.records) != 1:
                raise RuntimeError(
                    f"Expected 1 input, got "
                    f"{len(manifest.records)}"
                )

            processed = out_dir / "processed"

            writer = BoltzWriter(
                data_dir=processed / "structures",
                output_dir=out_dir / "predictions",
                output_format="mmcif",
                boltz2=True,
            )

            data_module = Boltz2InferenceDataModule(
                manifest=manifest,
                target_dir=processed / "structures",
                msa_dir=processed / "msa",
                mol_dir=MODEL_CACHE_DIR / "mols",
                num_workers=0,
                constraints_dir=(
                    processed / "constraints"
                    if (
                        processed / "constraints"
                    ).exists()
                    else None
                ),
                template_dir=(
                    processed / "templates"
                    if (
                        processed / "templates"
                    ).exists()
                    else None
                ),
                extra_mols_dir=(
                    processed / "mols"
                    if (
                        processed / "mols"
                    ).exists()
                    else None
                ),
                override_method=None,
            )

            preprocess_seconds = round(
                time.perf_counter()
                - preprocess_start,
                3,
            )

            print(
                f"[RESIDENT] Preprocessing: "
                f"{preprocess_seconds:.3f}s",
                flush=True,
            )

            # --------------------------------
            # Model inference
            # --------------------------------

            trainer = Trainer(
                default_root_dir=str(out_dir),
                accelerator="gpu",
                devices=1,
                precision="bf16-mixed",
                callbacks=[writer],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )

            before_device = str(
                next(
                    self.model.parameters()
                ).device
            )

            torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()

            trainer.predict(
                self.model,
                datamodule=data_module,
                return_predictions=False,
            )

            torch.cuda.synchronize()

            inference_seconds = round(
                time.perf_counter() - t0,
                3,
            )

            after_device = str(
                next(
                    self.model.parameters()
                ).device
            )

            # --------------------------------
            # Restore GPU residency if needed
            # --------------------------------

            restage_start = time.perf_counter()

            self.model.to("cuda")
            torch.cuda.synchronize()

            restage_seconds = round(
                time.perf_counter()
                - restage_start,
                3,
            )

            counts = _pred_counts(out_dir)

            if (
                counts["cif_count"] < 1
                or counts["confidence_count"] < 1
            ):
                raise RuntimeError(
                    "Resident prediction output "
                    f"missing: {counts}"
                )

            # --------------------------------
            # Save output and metrics
            # --------------------------------

            dest = (
                RESULTS_ROOT
                / run_id
                / "resident"
                / f"{self.counter:02d}-{ligand_id}"
            )

            shutil.copytree(
                out_dir / "predictions",
                dest / "predictions",
            )

            result = {
                "mode": "cached_python_model_object",
                "ligand": ligand_id,
                "request_index_in_container": (
                    self.counter
                ),
                "container_started_at": (
                    self.container_started_at
                ),
                "gpu": self.gpu_name,
                "init_seconds_once_per_container": (
                    self.init_seconds
                ),
                "initial_model_vram_mib": (
                    self.initial_vram_mb
                ),
                "model_device_before": before_device,
                "model_device_after_trainer": (
                    after_device
                ),
                "preprocessing_seconds": (
                    preprocess_seconds
                ),
                "trainer_predict_seconds": (
                    inference_seconds
                ),
                "gpu_restage_seconds": (
                    restage_seconds
                ),
                "total_method_seconds": round(
                    time.perf_counter()
                    - request_start,
                    3,
                ),
                "gpu_peak_memory_mib": round(
                    torch.cuda.max_memory_allocated()
                    / 2**20,
                    1,
                ),
                **counts,
            }

            (dest / "metrics.json").write_text(
                json.dumps(
                    result,
                    indent=2,
                ),
                encoding="utf-8",
            )

            results_volume.commit()

            print(
                f"[RESIDENT] Prediction "
                f"{self.counter}: "
                f"{result['total_method_seconds']:.3f}s",
                flush=True,
            )

            return result


# ============================================================
# Local benchmark entrypoint
# ============================================================

@app.local_entrypoint()
def benchmark(repeats: int = 2):

    if not 2 <= repeats <= 5:
        raise ValueError(
            "--repeats must be between 2 and 5"
        )

    import uuid

    run_id = (
        "bench-" + uuid.uuid4().hex[:10]
    )

    print(
        f"run_id={run_id}; "
        "GPU charges apply; "
        "input MSA=empty",
        flush=True,
    )

    # ------------------------------------
    # CLI benchmark
    # ------------------------------------

    cli_result = cli_benchmark.remote(
        run_id,
        repeats,
    )

    print("CLI results:")
    print(
        json.dumps(
            cli_result,
            indent=2,
            ensure_ascii=False,
        )
    )

    # ------------------------------------
    # Resident model benchmark
    # ------------------------------------

    worker = ResidentBoltz()
    resident_records = []

    for _ in range(repeats):

        name, smiles = BENCHMARK_LIGAND

        row = worker.predict.remote(
            run_id,
            name,
            smiles,
        )

        resident_records.append(row)

        print(
            "Resident:",
            json.dumps(
                row,
                ensure_ascii=False,
            ),
            flush=True,
        )

    same_container = (
        len({
            r["container_started_at"]
            for r in resident_records
        }) == 1
    )

    aggregate = {
        "run_id": run_id,
        "n": repeats,
        "cli": cli_result,
        "resident": resident_records,
        "resident_same_container": (
            same_container
        ),
        "resident_mean_method_seconds": round(
            sum(
                r["total_method_seconds"]
                for r in resident_records
            ) / repeats,
            3,
        ),
        "resident_steady_state_seconds": (
            [
                r["total_method_seconds"]
                for r in resident_records[1:]
            ]
            if same_container
            else []
        ),
        "warning": (
            "Model object reuse is tested; "
            "GPU memory residency is not presumed. "
            "Joltz is tested separately."
        ),
    }

    output_path = Path(
        f"boltz-cache-benchmark-{run_id}.json"
    )

    output_path.write_text(
        json.dumps(
            aggregate,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("FINAL BENCHMARK:")

    print(
        json.dumps(
            aggregate,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "Results Volume: "
        "py -m modal volume ls "
        f"opendock-boltz-benchmarks /{run_id}"
    )
