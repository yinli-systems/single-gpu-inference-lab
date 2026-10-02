"""Verify every original member; reproduce descriptive Native HTTP/SLO evidence."""

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "sgi-native-graph-http-20261002"
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ARCHIVE = "d152966077600b47d8f064283b33bbe294ffa0a9b559232b6a48f6425200292f"


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            result.update(data)
    return result.hexdigest()


def verify_extract(archive, receipt, destination):
    assert receipt["archive_sha256"] == EXPECTED_ARCHIVE
    assert sha(archive) == EXPECTED_ARCHIVE
    assert archive.stat().st_size == receipt["archive_bytes"]
    seen = set()
    with tarfile.open(archive) as stream:
        for member in stream:
            name = Path(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name in receipt["original_files"] and member.name not in seen
            raw = stream.extractfile(member).read()
            assert hashlib.sha256(raw).hexdigest() == receipt["original_files"][member.name]
            out = destination / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
            seen.add(member.name)
    assert seen == set(receipt["original_files"]) and len(seen) == 499
    binding = json.loads((destination / "binding.json").read_text())
    assert sha(destination / "harness.tar.gz") == binding["archive_sha256"]
    for name, digest in binding["source_files"].items():
        assert sha(destination / "harness" / name) == digest
    return len(seen)


def build(archive, receipt_path, output):
    assert not output.exists(), "Create a new output directory; retain earlier reproductions"
    receipt = json.loads(receipt_path.read_text())
    with tempfile.TemporaryDirectory(prefix="sgi-graph-http-proof-") as temporary:
        original = Path(temporary)
        members = verify_extract(archive, receipt, original)
        # Reuse the exact measurement dependencies retained inside the frozen
        # archive, and the separately reviewed CPU/GPU correlation auditor.
        sys.path.insert(0, str(original / "harness"))
        spec = importlib.util.spec_from_file_location(
            "independent_graph_proof",
            original / "independent-analysis-source/graph_serving_audit-ce23bbf.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        groups = {}
        audits = {}
        for gpu, job in (("RTX4090", "1650925"), ("RTX5090", "1650926")):
            run = original / "runs" / (("gpu_4090" if gpu == "RTX4090" else "gpu_5090") + "-" + job)
            audits[gpu] = module.observe(run)
            assert audits[gpu] == json.loads(
                (original / f"receipts/audit-ce23bbf-{job}.json").read_text()
            )
            assert (original / f"receipts/exit-{job}.txt").read_text().strip() == "0"
            for name in module.WORKLOADS:
                blocks = [
                    json.loads(
                        (run / "native_baseline/observations" / f"{name}-b{b}.json").read_text()
                    )
                    for b in range(4)
                ]
                requests = [r for block in blocks for r in block["requests"]]
                elapsed = sum(block["elapsed"] for block in blocks)
                itl = [
                    y - x for r in requests for x, y in zip(r["token_times"], r["token_times"][1:])
                ]
                groups[gpu, name] = {
                    "gpu": gpu,
                    "workload": name,
                    "complete_blocks": 4,
                    "requests": len(requests),
                    "output_tokens": sum(len(r["tokens"]) for r in requests),
                    "elapsed_sum_seconds": elapsed,
                    "requests_per_second": len(requests) / elapsed,
                    "output_tokens_per_second": sum(len(r["tokens"]) for r in requests) / elapsed,
                    "logical_input_tokens_per_second": sum(
                        len(r["payload"]["input_ids"]) for r in requests
                    )
                    / elapsed,
                    "uncached_input_tokens_per_second": sum(
                        len(r["payload"]["input_ids"]) - r["cached_tokens"] for r in requests
                    )
                    / elapsed,
                    **{
                        f"{k}_pooled_{label}_ms": float(np.quantile([r[k] for r in requests], q))
                        * 1000
                        for k in ("ttft", "tpot")
                        for label, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
                    },
                    "client_itl_pooled_p99_ms": float(np.quantile(itl, 0.99)) * 1000,
                    "co_delivered_zero_intervals": sum(v == 0 for v in itl),
                    "strict_goodput_ttft2s_tpot50ms": sum(
                        r["ttft"] <= 2 and r["tpot"] <= 0.05 for r in requests
                    )
                    / elapsed,
                    "slo_grid": [
                        [
                            sum(r["ttft"] <= ttft and r["tpot"] <= tpot for r in requests)
                            / len(requests)
                            for tpot in (0.005, 0.01, 0.02, 0.05, 0.1)
                        ]
                        for ttft in (0.1, 0.25, 0.5, 1, 2, 5)
                    ],
                }
    output.mkdir(parents=True)
    with (output / "native-http-baseline.csv").open("w", newline="") as stream:
        fields = [k for k in next(iter(groups.values())) if k != "slo_grid"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in groups.values():
            writer.writerow({k: value[k] for k in fields})
    for gpu in audits:
        fig, axes = plt.subplots(1, 5, figsize=(17, 4.5), sharey=True)
        fig.suptitle(f"{gpu}: complete Native HTTP / fixed SLO grid", fontsize=16, y=0.97)
        fig.text(
            0.5,
            0.895,
            "Qwen2.5-Coder-1.5B BF16 · ordinary CUDA Graph + overlap · one allocation · descriptive, no gain / CI claim",
            ha="center",
            fontsize=9,
        )
        for ax, name in zip(axes, module.WORKLOADS):
            z = np.array(groups[gpu, name]["slo_grid"])
            image = ax.imshow(z, origin="lower", vmin=0, vmax=1, cmap="Blues", aspect="auto")
            ax.set_title(name.replace("_", " "), fontsize=10)
            ax.set_xticks(range(5), ["5", "10", "20", "50", "100"])
            ax.set_yticks(range(6), ["100", "250", "500", "1000", "2000", "5000"])
            ax.set_xlabel("TPOT limit (ms)", fontsize=9)
            for row in range(6):
                for col in range(5):
                    ax.text(
                        col,
                        row,
                        f"{z[row, col]:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if z[row, col] > 0.65 else "#17283c",
                    )
        axes[0].set_ylabel("TTFT limit (ms)", fontsize=9)
        fig.subplots_adjust(left=0.055, right=0.90, bottom=0.19, top=0.79, wspace=0.15)
        color = fig.add_axes([0.92, 0.20, 0.014, 0.59])
        fig.colorbar(image, cax=color, label="Requests meeting BOTH limits")
        fig.text(
            0.055,
            0.065,
            "All 4 blocks pooled per workload; failed requests omitted: 0. Client SSE arrivals retain co-delivered zero intervals. Original canary: HOLD.",
            fontsize=9,
        )
        fig.savefig(output / f"slo-grid-{gpu}.png", dpi=180)
        fig.savefig(output / f"slo-grid-{gpu}.svg", metadata={"Date": None})
        plt.close(fig)
    result = {
        "verified_archive_sha256": EXPECTED_ARCHIVE,
        "verified_members": members,
        "harness_commit": receipt["harness_commit"],
        "independent_audit_commit": receipt["observer_audit_commit"],
        "audits": audits,
        "native_baseline_groups": list(groups.values()),
        "scope": "Two GPU families, one complete model, one allocation per GPU; full fixed 5-workload x 4-block HTTP matrix and descriptive SLO grid. Resource disabled. No offered-load sweep, independent-process confidence interval, performance gain or full HTTP qualification.",
        "original_canary": "HOLD",
        "full_http_qualified": False,
        "resource_graph_serving_qualified": False,
        "default_promotion": False,
        "serving_promotion": False,
        "historical_token_divergence_resolved": False,
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = "".join(
        f"<tr><td>{v['gpu']}</td><td>{v['workload']}</td><td>{v['requests']}</td><td>{v['output_tokens_per_second']:.1f}</td><td>{v['ttft_pooled_p99_ms']:.2f}</td><td>{v['tpot_pooled_p99_ms']:.2f}</td><td>{v['strict_goodput_ttft2s_tpot50ms']:.2f}</td></tr>"
        for v in groups.values()
    )
    html = (
        """<!doctype html>
<html lang="zh">
<meta charset="utf-8">
<title>
GraphServing / HTTP-SLO evidence</title>
<style>
body{max-width:1350px;margin:48px auto;padding:0 24px;background:#f5f7fb;color:#17283c;font:16px/1.7 system-ui}h1{font-size:32px}article{background:white;border:1px solid #dce3ed;border-radius:12px;padding:24px;margin:24px 0}img{width:100%}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e2e7ef}code{background:#edf2f8;padding:2px 5px}a{color:#1754b5}</style>
<h1>
GraphServing 与完整 HTTP/SLO 原始证据</h1>
<p>
已完成双卡、Qwen2.5-Coder-1.5B、5 类负载 × 4 个完整块。每卡单次 allocation，Resource 关闭；吞吐与尾延迟均为描述性基线，不具备收益、置信区间或完整 HTTP 资格。</p>
<article>
<b>
真实 Graph 证据：</b>
每卡 2,395 个实际输入 epoch，34 次独立关联 Graph launch、952 次注意力内核。4090：128 请求 / 7,936 token 一致；5090：9 / 128 请求存在差异，共 900 个 token 位置，诊断 HOLD。保留全部差异，不能归因或删除。<p>
历史 2/432 仍未 closure；<code>
default_promotion=false</code>
，<code>
serving_promotion=false</code>
。</p>
</article>
<article>
<h2>
普通 Native 基线</h2>
<p>
跨 4 个块按所有实际请求汇总分位数，吞吐分母为完整块耗时之和；仪器读回与 profile 数据不参与性能表。Goodput 固定 TTFT ≤ 2s、TPOT ≤ 50ms。</p>
<table>
<tr>
<th>
GPU</th>
<th>
负载</th>
<th>
请求</th>
<th>
输出 token/s</th>
<th>
TTFT p99 ms</th>
<th>
TPOT p99 ms</th>
<th>
Goodput 请求/s</th>
</tr>
"""
        + rows
        + """</table></article><article><img src="slo-grid-RTX4090.png" alt="4090 full SLO threshold grid"></article><article><img src="slo-grid-RTX5090.png" alt="5090 full SLO threshold grid"></article><p><a href="analysis.json">完整可复核 JSON</a> · <a href="native-http-baseline.csv">基线 CSV</a> · <a href="slo-grid-RTX4090.svg">4090 SVG</a> · <a href="slo-grid-RTX5090.svg">5090 SVG</a></p></html>"""
    )
    (output / "index.html").write_text(html)
    manifest = {
        "inputs": {
            str(archive.relative_to(ROOT)): sha(archive),
            str(receipt_path.relative_to(ROOT)): sha(receipt_path),
        },
        "generator_sha256": sha(Path(__file__)),
        "outputs": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "verified_members": members,
        "qualification_authority": False,
        "default_promotion": False,
        "serving_promotion": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verified_members": members,
                "states": {g: a["state"] for g, a in audits.items()},
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "evidence/v42-native-graph-serving-51be73c/raw-native-graph-http.tar.gz",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=ROOT / "evidence/v42-native-graph-serving-51be73c/receipt.json",
    )
    args = parser.parse_args()
    build(args.archive, args.receipt, args.output)
