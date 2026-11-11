
"""Boltz-2 CLI runner with profiling."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import threading
import time

from worker.contracts import Options


class BoltzExecutionError(RuntimeError):
    pass


def make_command(
    input_yaml: Path,
    output_dir: Path,
    cache_dir: Path,
    options: Options,
) -> list[str]:

    return [
        "boltz", "predict", str(input_yaml),
        "--model", "boltz2",
        "--out_dir", str(output_dir),
        "--cache", str(cache_dir),
        "--accelerator", "gpu",
        "--devices", "1",
        "--output_format", "mmcif",
        "--num_workers", "0",
        "--preprocessing-threads", "2",
        "--override",
        "--recycling_steps", str(options.recycling_steps),
        "--sampling_steps", str(options.sampling_steps),
        "--diffusion_samples", str(options.diffusion_samples),
    ]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


def _sample_gpu(
    stop: threading.Event,
    output_path: Path
) -> None:

    """Record GPU utilization and memory every 5 seconds."""

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow([
            "time_utc",
            "gpu_util_percent",
            "memory_used_mib",
            "memory_total_mib",
        ])

        handle.flush()

        while not stop.is_set():

            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=True,
                )

                values = [
                    value.strip()
                    for value in result.stdout.splitlines()[0].split(",")
                ]

                if len(values) == 3:
                    writer.writerow([
                        _timestamp(),
                        *values
                    ])
                    handle.flush()

            except (
                OSError,
                subprocess.SubprocessError,
                IndexError
            ):
                writer.writerow([
                    _timestamp(), "N/A", "N/A", "N/A"
                ])
                handle.flush()

            stop.wait(5)


def run_boltz(
    input_yaml: Path,
    output_dir: Path,
    cache_dir: Path,
    options: Options,
    *,
    timeout_seconds: int = 1500,
) -> Path:

    output_dir.mkdir(
        parents=True, exist_ok=True
    )

    cache_dir.mkdir(
        parents=True, exist_ok=True
    )

    log_path = output_dir.parent / "boltz.log"
    gpu_path = output_dir.parent / "gpu_usage.csv"

    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1"
    )

    stop = threading.Event()
    sampler = None
    process = None

    start = time.perf_counter()

    print(
        f"[PROFILE] Boltz CLI start: {_timestamp()}",
        flush=True
    )

    try:
        with log_path.open(
            "w",
            encoding="utf-8",
            buffering=1
        ) as log:

            process = subprocess.Popen(
                make_command(
                    input_yaml,
                    output_dir,
                    cache_dir,
                    options
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )

            def forward_output() -> None:

                assert process is not None
                assert process.stdout is not None

                for line in process.stdout:

                    value = line.rstrip("\r\n")

                    stamped = (
                        f"[{_timestamp()}] {value}"
                    )

                    log.write(stamped + "\n")
                    print(stamped, flush=True)

            reader = threading.Thread(
                target=forward_output,
                daemon=True
            )

            reader.start()

            sampler = threading.Thread(
                target=_sample_gpu,
                args=(stop, gpu_path),
                daemon=True
            )

            sampler.start()

            try:
                returncode = process.wait(
                    timeout=timeout_seconds
                )

            except subprocess.TimeoutExpired as exc:

                process.kill()
                process.wait()

                raise BoltzExecutionError(
                    "Boltz CLI timed out"
                ) from exc

            finally:
                reader.join(timeout=10)

    except OSError as exc:
        raise BoltzExecutionError(
            "Boltz CLI could not start"
        ) from exc

    finally:
        stop.set()

        if sampler is not None:
            sampler.join(timeout=10)

        elapsed = time.perf_counter() - start

        print(
            f"[PROFILE] Boltz CLI elapsed: {elapsed:.2f}s",
            flush=True
        )

    if returncode != 0:

        tail = log_path.read_text(
            encoding="utf-8",
            errors="replace"
        )[-6000:]

        raise BoltzExecutionError(
            f"Boltz exited with code {returncode}\n{tail}"
        )

    return output_dir
