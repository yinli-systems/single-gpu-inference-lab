"""Actual allocated GPU telemetry with UTC Unix bounds for HTTP block alignment."""

import argparse
import csv
import json
import subprocess
import time


def collect(uuid, output):
    with output.open("x", buffering=1) as stream:
        while True:
            started = time.time()
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=" + uuid,
                    "--query-gpu=uuid,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            finished = time.time()
            record = {
                "unix_started": started,
                "unix_finished": finished,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            try:
                rows = list(csv.reader(result.stdout.splitlines()))
                if result.returncode or len(rows) != 1 or len(rows[0]) != 6:
                    raise ValueError("One actual GPU row required")
                values = [v.strip() for v in rows[0]]
                if values[0] != uuid:
                    raise ValueError("Actual allocated GPU UUID differs")
                record.update(
                    uuid=uuid,
                    utilization=float(values[1]),
                    sm_mhz=float(values[2]),
                    memory_mhz=float(values[3]),
                    watts=float(values[4]),
                    temperature=float(values[5]),
                )
            except (ValueError, TypeError) as error:
                record["error"] = str(error)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            time.sleep(2)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("uuid")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    collect(args.uuid, args.output)
