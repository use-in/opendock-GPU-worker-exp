"""Invoke an already deployed Modal Function."""

import argparse
import json

from pathlib import Path

import modal


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "job_json",
        help="validated job JSON file"
    )

    args = parser.parse_args()

    job = json.loads(
        Path(args.job_json).read_text(
            encoding="utf-8"
        )
    )

    fn = modal.Function.from_name(
        "opendock-boltz2-worker-prod",
        "predict_s3"
    )

    result = fn.remote(job)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()