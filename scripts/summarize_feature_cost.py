#!/usr/bin/env python3
"""Aggregate measure_vllm_feature_cost.py runs into an artifact summary.

Reads one or more raw run files (each already carries per-round results and a
per-condition median summary), an optional set of equivalence JSONs from
compare_feature_cost_outputs.py, and writes ``summary.json`` plus a Markdown
table block that the artifact README embeds verbatim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def condition_rows(run: dict) -> list[dict]:
    rows = []
    base = run["summary"].get("native/gen")
    for key, val in run["summary"].items():
        server, rc = key.split("/")
        rounds = [r for r in run["servers"][server]["rounds"] if r["request_condition"] == rc]
        tps = [r["output_tokens_per_s"] for r in rounds]
        e2e = [r["e2e_ms"]["median"] for r in rounds]
        p99 = [r["e2e_ms"]["p99"] for r in rounds]
        row = {
            "server": server,
            "request": rc,
            "rounds": len(rounds),
            "output_tokens_per_round": rounds[0]["output_tokens"],
            "tok_per_s_median": statistics.median(tps),
            "tok_per_s_min": min(tps),
            "tok_per_s_max": max(tps),
            "e2e_ms_median": statistics.median(e2e),
            "e2e_ms_p99_median": statistics.median(p99),
            "mask_entries": rounds[0]["mask_entries_total"],
            "mask_mean_size": rounds[0]["mask_mean_size"],
            "mask_max_size": rounds[0]["mask_max_size"],
            "body_bytes": rounds[0]["body_bytes_total"],
            "tokens_per_chunk": rounds[0].get("tokens_per_chunk"),
            "flashinfer_sampler_selected": run["servers"][server]
            .get("log_markers", {})
            .get("flashinfer_sampler_selected"),
        }
        if base:
            row["tok_per_s_vs_native_gen"] = row["tok_per_s_median"] / base["output_tokens_per_s_median"]
        rows.append(row)
    return rows


def md_table(rows: list[dict], with_itl: bool) -> str:
    head = "| Server | Request | tok/s (median, min–max) | vs native/gen | e2e median | e2e p99 | mask mean size | FlashInfer sampler |"
    sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"
    out = [head, sep]
    for r in rows:
        mask = f"{r['mask_mean_size']:.1f}" if r["mask_mean_size"] else "—"
        fi = {True: "yes", False: "no", None: "?"}[r["flashinfer_sampler_selected"]]
        out.append(
            f"| `{r['server']}` | `{r['request']}` | {r['tok_per_s_median']:,.0f} ({r['tok_per_s_min']:,.0f}–{r['tok_per_s_max']:,.0f}) "
            f"| {r.get('tok_per_s_vs_native_gen', float('nan')):.3f}x | {r['e2e_ms_median']:,.0f} ms | {r['e2e_ms_p99_median']:,.0f} ms | {mask} | {fi} |"
        )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path to a raw run json")
    ap.add_argument("--equivalence", action="append", default=[], help="label=path to a comparer json")
    ap.add_argument("--artifact-dir", type=Path, required=True)
    args = ap.parse_args()

    summary = {"schema_version": 1, "result_type": "vllm_sampling_mask_serving_ab", "runs": {}, "equivalence": {}}
    md = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        run = load(Path(path))
        rows = condition_rows(run)
        summary["runs"][label] = {
            "raw_file": str(Path(path).name),
            "raw_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "model": run["model"],
            "workload": run["workload"],
            "environment": run["environment"],
            "provenance": run["provenance"],
            "servers": {
                name: {"flags": s["flags"], "env": s.get("env", {}), "log_markers": s.get("log_markers")}
                for name, s in run["servers"].items()
            },
            "rows": rows,
        }
        md.append(f"### {label}\n\nModel `{Path(run['model']).name}`, {run['workload']['prompts']} prompts x {run['workload']['max_tokens']} tokens (ignore_eos), concurrency {run['workload']['concurrency']}, API `{run['workload'].get('api', 'completions')}`, sampling {run['workload']['sampling']}, {run['workload']['rounds']} interleaved rounds.\n")
        md.append(md_table(rows, with_itl=run["workload"].get("api") == "completions"))
        md.append("")
    for spec in args.equivalence:
        label, path = spec.split("=", 1)
        summary["equivalence"][label] = load(Path(path))
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.artifact_dir / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    if summary["equivalence"]:
        print(json.dumps(summary["equivalence"], indent=2))


if __name__ == "__main__":
    main()
