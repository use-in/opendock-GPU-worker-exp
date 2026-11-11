
"""Independent Joltz experimental probe.

Run:
  py -m modal run modal_joltz_probe.py::inspect
  py -m modal run modal_joltz_probe.py::run_example

Stage 1: Check Joltz, JAX and GPU availability.
Stage 2: Execute the upstream Joltz example.py.

This is NOT a production OpenDock worker.
"""
from __future__ import annotations

import json
import modal

app = modal.App(
    "opendock-joltz-experiment"
)

image = (
    modal.Image.debian_slim(
        python_version="3.12"
    )
    .apt_install(
        "git",
        "libgomp1",
        "libgl1",
    )
    .pip_install(
        "jax[cuda12]",
        "git+https://github.com/nboyd/joltz.git",
    )
    .run_commands(
        "git clone --depth 1 "
        "https://github.com/nboyd/joltz.git "
        "/opt/joltz"
    )
)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=900,
    max_containers=1,
)
def probe_environment() -> dict:

    import importlib.metadata
    import pkgutil
    import subprocess
    from pathlib import Path

    import jax
    import joltz

    revision = subprocess.run(
        [
            "git",
            "-C",
            "/opt/joltz",
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    devices = [
        str(device)
        for device in jax.devices()
    ]

    return {
        "joltz_repo_commit": revision,
        "joltz_dist_version": (
            importlib.metadata.version("joltz")
        ),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "devices": devices,
        "joltz_modules": sorted(
            module.name
            for module in pkgutil.iter_modules(
                joltz.__path__
            )
        ),
        "example_present": (
            Path("/opt/joltz/example.py").is_file()
        ),
        "limitation": (
            "This checks installation and GPU access, "
            "not full Boltz-2 inference compatibility."
        ),
    }


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=1200,
    max_containers=1,
)
def execute_example() -> dict:

    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path("/opt/joltz")
    example = root / "example.py"

    if not example.exists():
        raise FileNotFoundError(
            "Joltz example.py does not exist"
        )

    started = time.perf_counter()

    process = subprocess.run(
        [
            sys.executable,
            str(example),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=900,
    )

    elapsed = round(
        time.perf_counter() - started,
        3,
    )

    return {
        "exit_code": process.returncode,
        "seconds": elapsed,
        "stdout_tail": process.stdout[-12000:],
        "stderr_tail": process.stderr[-12000:],
        "interpretation": (
            "Success means the upstream example runs. "
            "It does not prove full OpenDock "
            "Boltz-2 feature compatibility."
        ),
    }


@app.local_entrypoint()
def inspect():

    result = probe_environment.remote()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


@app.local_entrypoint()
def run_example():

    result = execute_example.remote()

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    if result["exit_code"] != 0:
        raise RuntimeError(
            "Joltz upstream example failed. "
            "Check stdout_tail and stderr_tail."
        )
