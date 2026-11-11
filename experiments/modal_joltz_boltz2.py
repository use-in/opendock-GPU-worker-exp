
"""
OpenDock - Boltz-2 + Joltz JAX Compatibility Benchmark

Boltz: 2.2.1
Joltz: 0.1.0 (pinned commit)
GPU: NVIDIA H100
Input: Ubiquitin + Ethanol
MSA: empty
Batch: disabled

Commands:
    py -m modal run modal_joltz_boltz2.py::inspect
    py -m modal run modal_joltz_boltz2.py::api
    py -m modal run modal_joltz_boltz2.py::convert
    py -m modal run modal_joltz_boltz2.py::predict

Experimental compatibility test.
Not intended for production deployment.
"""

from __future__ import annotations

import json
import time
import traceback
import gc

from pathlib import Path
from dataclasses import asdict
from tempfile import TemporaryDirectory
from importlib.metadata import version

import modal


# ============================================================
# Configuration
# ============================================================

BOLTZ_VERSION = "2.2.1"

JOLTZ_COMMIT = (
    "ed0f04257dac85bd4b7bf45521cc280f86d38ded"
)

MODEL_CACHE = Path("/model-cache/boltz")

PROTEIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLED"
    "GRTLSDYNIQKESTLHLVLRLRGG"
)

LIGAND_SMILES = "CCO"

RECYCLING_STEPS = 1
SAMPLING_STEPS = 20
RANDOM_SEED = 42


# ============================================================
# Modal Setup
# ============================================================

app = modal.App(
    "opendock-joltz-boltz2-compat"
)

volume = modal.Volume.from_name(
    "opendock-boltz-model-cache",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .apt_install(
        "git",
        "libgomp1",
        "libgl1",
    )
    .pip_install(
        "jax[cuda12]==0.7.1",
        (
            "git+https://github.com/nboyd/joltz.git@"
            f"{JOLTZ_COMMIT}"
        ),
    )
    .pip_install(
        f"boltz[cuda]=={BOLTZ_VERSION}"
    )
    .env({
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONUNBUFFERED": "1",
    })
)


# ============================================================
# 1. Environment Check
# ============================================================

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=180,
    retries=0,
)
def check_versions():

    import jax
    import joltz
    import torch

    return {
        "boltz_version": version("boltz"),
        "joltz_version": version("joltz"),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "torch_cuda_available": (
            torch.cuda.is_available()
        ),
        "translator_available": callable(
            getattr(joltz, "from_torch", None)
        ),
        "note": (
            "This function runs without GPU. "
            "CPU backend is expected."
        ),
    }


# ============================================================
# 2. Joltz2 API Inspection
# ============================================================

@app.function(
    image=image,
    cpu=2,
    memory=4096,
    timeout=180,
    retries=0,
)
def inspect_joltz2_api():

    import inspect
    import joltz

    model_class = getattr(
        joltz,
        "Joltz2",
        None,
    )

    if model_class is None:
        return {
            "status": "failed",
            "error": "Joltz2 class not found",
        }

    method = model_class.__call__

    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        source = "Source unavailable"

    return {
        "status": "success",
        "class_name": model_class.__name__,
        "class_module": model_class.__module__,
        "call_signature": str(
            inspect.signature(method)
        ),
        "call_source": source[:12000],
    }


# ============================================================
# 3. Load Boltz-2 PyTorch Model
# ============================================================

def load_boltz2():

    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
        download_boltz2,
    )

    from boltz.model.models.boltz2 import Boltz2

    MODEL_CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    download_boltz2(MODEL_CACHE)

    checkpoint = (
        MODEL_CACHE / "boltz2_conf.ckpt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}"
        )

    predict_args = {
        "recycling_steps": RECYCLING_STEPS,
        "sampling_steps": SAMPLING_STEPS,
        "diffusion_samples": 1,
        "max_parallel_samples": None,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    kwargs = {
        "strict": True,
        "map_location": "cpu",
        "ema": False,
        "predict_args": predict_args,
        "diffusion_process_args": asdict(
            Boltz2DiffusionParams()
        ),
        "pairformer_args": asdict(
            PairformerArgsV2()
        ),
        "msa_args": asdict(
            MSAModuleArgs(
                subsample_msa=False,
                use_paired_feature=True,
            )
        ),
        "steering_args": asdict(
            BoltzSteeringParams()
        ),
    }

    # Prefer PyTorch reference implementations
    # for compatibility with Joltz conversion.

    try:
        model = Boltz2.load_from_checkpoint(
            checkpoint,
            use_kernels=False,
            **kwargs,
        )

        kernel_mode = "torch_reference"

    except Exception as exc:

        print(
            f"[WARN] Reference mode failed: {exc}",
            flush=True,
        )

        model = Boltz2.load_from_checkpoint(
            checkpoint,
            use_kernels=True,
            **kwargs,
        )

        kernel_mode = "cuda_kernels"

    model.eval()

    return model, kernel_mode


# ============================================================
# 4. Prepare Boltz-2 Input Features
# ============================================================

def make_jax_features():

    import jax
    import jax.numpy as jnp
    import torch
    import yaml

    from rdkit import Chem

    from boltz.main import process_inputs
    from boltz.data.types import Manifest

    from boltz.data.module.inferencev2 import (
        Boltz2InferenceDataModule,
    )

    with TemporaryDirectory(
        prefix="joltz-boltz2-"
    ) as temp:

        root = Path(temp)

        # ----------------------------------------
        # Create Boltz input YAML
        # ----------------------------------------

        input_file = root / "complex.yaml"

        input_data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": PROTEIN,
                        "msa": "empty",
                    }
                },
                {
                    "ligand": {
                        "id": "B",
                        "smiles": LIGAND_SMILES,
                    }
                },
            ],
        }

        input_file.write_text(
            yaml.safe_dump(input_data),
            encoding="utf-8",
        )

        # ----------------------------------------
        # Preprocessing
        # ----------------------------------------

        output_dir = (
            root / "boltz_results_complex"
        )

        output_dir.mkdir()

        Chem.SetDefaultPickleProperties(
            Chem.PropertyPickleOptions.AllProps
        )

        process_inputs(
            data=[input_file],
            out_dir=output_dir,
            ccd_path=MODEL_CACHE / "ccd.pkl",
            mol_dir=MODEL_CACHE / "mols",
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

        processed = (
            output_dir / "processed"
        )

        manifest = Manifest.load(
            processed / "manifest.json"
        )

        if len(manifest.records) != 1:
            raise RuntimeError(
                "Expected exactly one input record"
            )

        # ----------------------------------------
        # Create inference DataModule
        # ----------------------------------------

        data_module = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=(
                processed / "structures"
            ),
            msa_dir=(
                processed / "msa"
            ),
            mol_dir=(
                MODEL_CACHE / "mols"
            ),
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

        data_module.setup(
            stage="predict"
        )

        dataloader = (
            data_module.predict_dataloader()
        )

        batch = next(
            iter(dataloader)
        )

        features = (
            batch.get("feats", batch)
            if isinstance(batch, dict)
            else batch
        )

        if not isinstance(features, dict):
            raise TypeError(
                "Expected a feature dictionary"
            )

        # ----------------------------------------
        # Convert PyTorch -> JAX arrays
        # ----------------------------------------

        def convert_tensor(value):

            if isinstance(value, torch.Tensor):
                return jnp.asarray(
                    value.detach().cpu().numpy()
                )

            return value

        jax_features = (
            jax.tree_util.tree_map(
                convert_tensor,
                features,
            )
        )

        return jax_features


# ============================================================
# 5. Synchronize GPU Execution
# ============================================================

def wait_for_gpu(tree):

    import jax

    leaves = (
        jax.tree_util.tree_leaves(tree)
    )

    for leaf in leaves:

        if hasattr(
            leaf,
            "block_until_ready",
        ):
            leaf.block_until_ready()


# ============================================================
# 6. Boltz-2 -> Joltz2 -> JAX Benchmark
# ============================================================

@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={
        str(MODEL_CACHE.parent): volume
    },
)
def run_trial(
    do_predict: bool = False
):

    report = {
        "boltz": None,
        "prediction_requested": do_predict,
        "recycling_steps": RECYCLING_STEPS,
        "sampling_steps": SAMPLING_STEPS,
        "msa": "empty",
        "ligand": LIGAND_SMILES,
    }

    stage = "imports"

    try:

        import jax
        import joltz

        report["boltz"] = version("boltz")

        report["joltz"] = version("joltz")

        report["jax"] = jax.__version__

        report["jax_backend"] = (
            jax.default_backend()
        )

        if report["boltz"] != BOLTZ_VERSION:
            raise RuntimeError(
                "Incorrect Boltz version: "
                f"{report['boltz']}"
            )

        if report["jax_backend"] != "gpu":
            raise RuntimeError(
                "JAX GPU backend unavailable"
            )

        # ========================================
        # Step 1: Load Boltz-2
        # ========================================

        stage = "load_checkpoint"

        print(
            "[1/3] Loading Boltz-2 checkpoint",
            flush=True,
        )

        start = time.perf_counter()

        model, kernel_mode = load_boltz2()

        report["kernel_mode"] = kernel_mode

        report["load_seconds"] = round(
            time.perf_counter() - start,
            3,
        )

        # ========================================
        # Step 2: Convert to Joltz2
        # ========================================

        stage = "joltz.from_torch"

        print(
            "[2/3] Converting Boltz-2 to Joltz2",
            flush=True,
        )

        start = time.perf_counter()

        jax_model = joltz.from_torch(
            model
        )

        report["convert_seconds"] = round(
            time.perf_counter() - start,
            3,
        )

        report["jax_model_type"] = (
            type(jax_model).__name__
        )

        print(
            "[SUCCESS] Model converted: "
            f"{report['jax_model_type']}",
            flush=True,
        )

        del model
        gc.collect()

        # Stop if only conversion was requested.

        if not do_predict:

            report["status"] = (
                "conversion_only"
            )

            return report

        # ========================================
        # Step 3: Prepare Input
        # ========================================

        stage = "prepare_features"

        print(
            "[3/3] Preparing Ubiquitin + ethanol",
            flush=True,
        )

        start = time.perf_counter()

        features = make_jax_features()

        report["preprocess_seconds"] = round(
            time.perf_counter() - start,
            3,
        )

        report["feature_keys"] = sorted(
            features.keys()
        )

        # ========================================
        # Step 4: Correct Joltz2 Forward API
        # ========================================

        # Exact API confirmed by inspect.signature:
        #
        # __call__(
        #     self,
        #     feats,
        #     recycling_steps=0,
        #     *,
        #     num_sampling_steps,
        #     deterministic,
        #     key,
        # )
        #
        # Joltz2 automatically computes:
        # - distance prediction
        # - structure sampling
        # - confidence metrics
        #
        # Do NOT pass sample_structure=True.

        key = jax.random.PRNGKey(
            RANDOM_SEED
        )

        def forward(
            feats,
            rng_key,
        ):

            return jax_model(
                feats,
                recycling_steps=(
                    RECYCLING_STEPS
                ),
                num_sampling_steps=(
                    SAMPLING_STEPS
                ),
                deterministic=True,
                key=rng_key,
            )

        # ========================================
        # Step 5: JAX Eager Forward
        # ========================================

        stage = "jax_eager_forward"

        print(
            "[JAX] Starting eager forward",
            flush=True,
        )

        start = time.perf_counter()

        result = forward(
            features,
            key,
        )

        # GPU operations are asynchronous.
        # Synchronize before recording time.

        wait_for_gpu(result)

        report["eager_seconds"] = round(
            time.perf_counter() - start,
            3,
        )

        if (
            not isinstance(result, tuple)
            or len(result) != 3
        ):
            raise RuntimeError(
                "Unexpected Joltz2 output. "
                "Expected (pdistogram, structure, "
                "confidence)."
            )

        report["eager_output_types"] = [
            type(item).__name__
            for item in result
        ]

        print(
            "[JAX] Eager forward completed: "
            f"{report['eager_seconds']}s",
            flush=True,
        )

        del result

        # ========================================
        # Step 6: JAX JIT Forward
        # ========================================

        stage = "jax_jit_forward"

        print(
            "[JAX] Compiling JIT function",
            flush=True,
        )

        # Fixed sampling configuration is captured
        # inside forward(), so no static_argnames
        # are required.

        jit_forward = jax.jit(
            forward
        )

        report["jit_seconds"] = []

        for trial in range(2):

            print(
                "[JIT] Starting trial "
                f"{trial + 1}/2",
                flush=True,
            )

            start = time.perf_counter()

            result = jit_forward(
                features,
                key,
            )

            wait_for_gpu(result)

            duration = round(
                time.perf_counter() - start,
                3,
            )

            report["jit_seconds"].append(
                duration
            )

            leaves = (
                jax.tree_util.tree_leaves(
                    result
                )
            )

            if not leaves:
                raise RuntimeError(
                    "JAX inference returned "
                    "no output leaves"
                )

            report["output_shapes"] = [
                list(item.shape)
                for item in leaves
                if hasattr(item, "shape")
            ][:20]

            print(
                f"[JIT] Trial {trial + 1}: "
                f"{duration:.3f}s",
                flush=True,
            )

            del result

        # ========================================
        # Step 7: Final Result
        # ========================================

        report["status"] = (
            "jax_forward_succeeded"
        )

        report["warning"] = (
            "JAX forward succeeded, but CIF output, "
            "confidence fidelity, and numerical parity "
            "with the PyTorch Boltz-2 implementation "
            "are not yet verified."
        )

        return report

    except Exception as exc:

        report.update({
            "status": "failed",
            "failed_stage": stage,
            "error_type": (
                type(exc).__name__
            ),
            "error": str(exc)[:2000],
            "traceback_tail": (
                traceback.format_exc()[-5000:]
            ),
        })

        print(
            "[FAILED] Stage: "
            f"{stage}",
            flush=True,
        )

        print(
            f"[FAILED] Error: {exc}",
            flush=True,
        )

        return report

    finally:
        volume.commit()


# ============================================================
# 7. Modal Local Entrypoints
# ============================================================

@app.local_entrypoint()
def inspect():

    result = check_versions.remote()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


@app.local_entrypoint()
def api():

    result = inspect_joltz2_api.remote()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


@app.local_entrypoint()
def convert():

    result = run_trial.remote(
        False
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


@app.local_entrypoint()
def predict():

    result = run_trial.remote(
        True
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
