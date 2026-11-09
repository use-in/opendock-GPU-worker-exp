"""Cloud-independent Boltz-2 CLI adapter."""

from __future__ import annotations

from pathlib import Path
import subprocess

from worker.contracts import Options


class BoltzExecutionError(RuntimeError):
    pass


def make_command(
    input_yaml: Path,
    output_dir: Path,
    cache_dir: Path,
    options: Options
) -> list[str]:

    return [
        "boltz",
        "predict",
        str(input_yaml),

        "--model",
        "boltz2",

        "--out_dir",
        str(output_dir),

        "--cache",
        str(cache_dir),

        "--accelerator",
        "gpu",

        "--devices",
        "1",

        "--output_format",
        "mmcif",

        "--num_workers",
        "0",

        "--preprocessing-threads",
        "2",

        "--override",

        "--recycling_steps",
        str(options.recycling_steps),

        "--sampling_steps",
        str(options.sampling_steps),

        "--diffusion_samples",
        str(options.diffusion_samples)
    ]


def run_boltz(
    input_yaml: Path,
    output_dir: Path,
    cache_dir: Path,
    options: Options,
    *,
    timeout_seconds: int = 1500
) -> Path:

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log = output_dir.parent / "boltz.log"

    with log.open("w", encoding="utf-8") as handle:

        try:
            process = subprocess.run(
                make_command(
                    input_yaml,
                    output_dir,
                    cache_dir,
                    options
                ),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds
            )

        except subprocess.TimeoutExpired as exc:
            raise BoltzExecutionError(
                "Boltz inference timed out; review Modal logs"
            ) from exc

        except OSError as exc:
            raise BoltzExecutionError(
                "Boltz executable could not be started"
            ) from exc

    if process.returncode != 0:
        raise BoltzExecutionError(
            f"Boltz exited with code {process.returncode}; "
            "inspect Modal logs"
        )

    return output_dir