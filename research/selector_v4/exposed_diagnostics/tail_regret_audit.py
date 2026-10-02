"""Read immutable archives; diagnose available-arm regret without changing policy.

No CUDA, scheduler, retries, threshold search, holdout training or authorization.
Run with python -m research.selector_v4.exposed_diagnostics.tail_regret_audit OUT.
"""

import argparse
import hashlib
import html
import json
import math
import tarfile
from pathlib import Path

import numpy as np

from research.selector_v4.exposed_diagnostics.public_path_contract import digest


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified_json(raw, expected):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Immutable input hash mismatch")
    return json.loads(raw)


def arm_means(rows):
    if len(rows) != 144:
        raise ValueError("All 24 complete six-window blocks required")
    arms = ("native", "oracle", "policy")
    for block in range(24):
        first = list(arms[block % 3 :] + arms[: block % 3])
        for position, (row, arm) in enumerate(
            zip(rows[block * 6 : (block + 1) * 6], first + first[::-1])
        ):
            if not (
                row["arm"] == arm
                and row["block"] == block
                and row["position"] == position
                and row["exact"]
                and row["elapsed_us"] >= 120000
                and row["kernel_calls"] > 0
                and row["wall_us_per_call"] > 0
                and math.isfinite(row["wall_us_per_call"])
                and math.isclose(
                    row["elapsed_us"] / row["kernel_calls"],
                    row["wall_us_per_call"],
                    rel_tol=1e-12,
                )
            ):
                raise ValueError("Original complete balanced windows required")
    return {
        arm: math.exp(
            float(np.log([r["wall_us_per_call"] for r in rows if r["arm"] == arm]).mean())
        )
        for arm in arms
    }


def certificate_detail(training):
    c = training["certificate"]
    if c["checksum"] != digest({k: v for k, v in c.items() if k != "checksum"}):
        raise ValueError("Certificate checksum mismatch")
    samples = np.asarray(c["samples"], dtype=float)
    if samples.shape != (32, 4) or not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("All original certificate samples required")
    seed = int(hashlib.sha256(c["identity"].encode()).hexdigest()[:8], 16)
    indices = np.random.default_rng(seed).integers(0, 32, size=(20000, 32))
    controls = [samples[:, 0] / samples[:, 3], samples[:, 1] / samples[:, 2]]
    cis = [np.exp(np.quantile(np.log(v)[indices].mean(axis=1), [0.05, 0.95])) for v in controls]
    if not np.allclose(cis, c["control_ci90"], rtol=1e-12, atol=0):
        raise ValueError("Raw certificate control intervals differ")
    passed = all(lo >= 1 / 1.005 and hi <= 1.005 for lo, hi in cis)
    if passed != c["checks"]["duplicate_controls"]:
        raise ValueError("Certificate control verdict differs")
    return {
        "accepted": training["certificate_accepted"],
        "managed_tactic": training["managed_tactic"],
        "failed_checks": [k for k, v in c["checks"].items() if not v],
        "control_ci90": c["control_ci90"],
        "native_control_upper_excess_basis_points": (cis[0][1] - 1.005) * 10000,
        "gain_ci95": c["gain_ci95"],
        "samples_retained": len(samples),
        "checksum": c["checksum"],
    }


def build(repo, out):
    receipt_path = (
        repo / "evidence/v42-public-completion-reaudit-77fa86f/corrected-accepted-receipt.json"
    )
    receipt = json.loads(receipt_path.read_text())
    index = json.loads((receipt_path.parent / "receipt.json").read_text())
    if sha(receipt_path) != index["accepted_receipt_sha256"]:
        raise ValueError("Accepted receipt hash mismatch")
    files = receipt["files"]
    names = {
        n
        for n in files
        if n in ("binding.json", "receipts/jobs.json")
        or n.startswith("analysis/gpu_")
        and n.endswith("/summary.json")
        or n.startswith("runs/")
        and (
            "/policy-" in n
            and n.endswith(
                ("/windows.json", "/frozen-choice-before-scoring.json", "/complete.json")
            )
            or "/train-" in n
            and n.endswith("/complete.json")
        )
    }
    data = {}
    archives = [
        (
            repo
            / "evidence/v42-public-complete-r2-0a5c735-hold/raw-public-path-development.tar.gz",
            index["original_hold_archive_sha256"],
        ),
        (receipt_path.parent / "new-independent-evidence.tar.gz", index["delta_archive_sha256"]),
    ]
    for archive, expected in archives:
        if sha(archive) != expected:
            raise ValueError("Original archive hash mismatch")
        seen = set()
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if member.name not in names:
                    continue
                if not member.isfile() or member.name in seen:
                    raise ValueError("Duplicate/nonregular evidence member")
                seen.add(member.name)
                data[member.name] = verified_json(
                    tar.extractfile(member).read(), files[member.name]
                )
    if set(data) != names:
        raise ValueError("Missing required original inputs")
    cards = {}
    for gpu, jobs in receipt["jobs"].items():
        summary = data[f"analysis/{gpu}/summary.json"]
        records = []
        for old in summary["records"]:
            case, key, rep = old["case"], old["key"], old["rep"]
            run = f"runs/{gpu}-{jobs[str(case)]}"
            phase = f"{run}/policy-{rep}/{key}"
            decision = data[f"{phase}/frozen-choice-before-scoring.json"]
            if decision["checksum"] != digest(
                {k: v for k, v in decision.items() if k != "checksum"}
            ):
                raise ValueError("Frozen choice checksum mismatch")
            training = {i: data[f"{run}/train-{i}/{key}/complete.json"] for i in range(3)}
            for i in decision["training_processes"]:
                if digest(training[i]) != decision["training_artifact_sha256"][str(i)]:
                    raise ValueError("Frozen training binding mismatch")
            means = arm_means(data[f"{phase}/windows.json"])
            regret = means["policy"] / min(means.values()) - 1
            primary = training[decision["primary_artifact"]]
            record = {
                "case": case,
                "key": key,
                "rep": rep,
                "dtype": old["dtype"],
                "layout": old["layout"],
                "execution": old["execution"],
                "choice": decision["choice"],
                "available_arm_regret": regret,
                "oracle_includes_resource": primary["managed_tactic"] == 1,
                "arm_geomean_us": means,
                "actual_policy_over_pristine_speedup": old["actual_policy_over_pristine_speedup"],
                "failed_selection_checks": [k for k, v in decision["checks"].items() if not v],
                "training_processes": decision["training_processes"],
                "decision_source": f"{phase}/frozen-choice-before-scoring.json",
            }
            if regret > 0.01:
                record["training_certificates"] = {
                    str(i): certificate_detail(training[i]) for i in range(3)
                }
            records.append(record)
        regrets = [r["available_arm_regret"] for r in records]
        metrics = {
            "p99": float(np.quantile(regrets, 0.99)),
            "worst": max(regrets),
            "above_one_percent_count": sum(r > 0.01 for r in regrets),
        }
        for key, value in metrics.items():
            if not math.isclose(
                value, summary["metrics"]["regret"][key], rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("Recomputed regret differs from accepted analysis")
        cards[gpu] = {
            "records": records,
            "metrics": metrics,
            "resource_oracle_records": sum(r["oracle_includes_resource"] for r in records),
            "complete_records": len(records),
            "original_development_pass": summary["pass"],
        }
        if len(records) != 144:
            raise ValueError("All 144 records per card required")
    report = {
        "scope": "RETROSPECTIVE_DIAGNOSIS_ONLY",
        "accepted_receipt_sha256": sha(receipt_path),
        "source_sha256": sha(Path(__file__)),
        "input_files": {n: files[n] for n in sorted(names)},
        "cards": cards,
        "all_original_archives_sha256_verified": True,
        "certificate_control_interval_comparison_rtol": 1e-12,
        "comparison_tolerance_does_not_change_gates": True,
        "default_promotion": False,
        "serving_promotion": False,
        "fresh_cases_consumed": 0,
        "gpu_jobs_dispatched": 0,
        "measured_optimization_gain": None,
        "historical_token_divergence_resolved": False,
        "physical_cause_of_certificate_instability": "UNRESOLVED",
        "interpretation": "Certificate duplicate-control failures force Native. Available-arm regret does not bound unmeasured Resource opportunities. No threshold changes or counterfactual speedup claims.",
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    tails = [
        r for card in cards.values() for r in card["records"] if r["available_arm_regret"] > 0.01
    ]
    table = "".join(
        f"<tr><td>{r['case']}</td><td>{r['dtype']}</td><td>{r['rep']}</td>"
        f"<td>{100 * r['available_arm_regret']:.4f}%</td><td>{r['choice']}</td>"
        f"<td>{html.escape(', '.join(r['failed_selection_checks']))}</td></tr>"
        for r in tails
    )
    page = (
        """<!doctype html><meta charset="utf-8"><title>4090 tail regret evidence audit</title>
<style>body{font:17px/1.6 system-ui;max-width:1100px;margin:48px auto;padding:0 24px;color:#162532}table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ccd6dd;text-align:left}code{background:#eef2f5}a{color:#145ea8}</style>
<h1>4090 tail regret：原始证据归因</h1>
<p>288条双卡记录全部纳入；原始归档与所用文件SHA256核验。没有新GPU测量、重试、删窗口或更改冻结策略。</p>
<p>三个超过1%的记录全部为eager_full_call。直接触发链为：训练证书Native重复控制区间超出1.005 → 证书拒绝 → managed Resource资格缺失 → 两进程选择规则回退Native。收益检查本身通过。</p>
<table><tr><th>Case</th><th>Dtype</th><th>Rep</th><th>Available-arm regret</th><th>Choice</th><th>Failed gates</th></tr>"""
        + table
        + """</table>
<p>这些来自两个已暴露case，是三个评分进程记录，并非三个独立新case。Native回退满足本轮既定安全门槛；不能据此声称生产安全已证明。</p>
<p>全部候选并非每次都可用。Resource未获得证书时，oracle臂也可能执行Native；低available-arm regret不能证明全候选最优。</p>
<h2>后续独立开发协议</h2><p>先研究证书测量的稳定性：在证书采样之前执行固定、预声明的Native条件化，并记录逐窗口控制漂移与遥测。所有32个证书块保留，控制区间门槛仍为[1/1.005, 1.005]，每次只有一次证书判决，失败保持失败。该方向只是待测假设；既有30秒条件化发生在后续评分前，不能替代证书前条件化。新协议应单独冻结，不能改写当前v4.3测量。</p>
<p>更强P99目标必须先明确候选覆盖与统计单位；缺失Resource候选应标记证据不足，不能记为零遗憾。当前报告不产生可执行证书，不降低门槛，不利用held-out oracle重选。</p>
<p>物理原因仍未确定；没有证明冷启动、时钟、温度或调度中的任何一个是根因。完整Resource SGLang GraphServing及HTTP资格仍待完成；历史2/432与首次普通并发差异仍未闭合。</p>
<p>现有72次HTTP协议只允许eager Resource与Native捕获图共存，明确拒绝captured Resource kernel。该流程即使通过，也不能关闭真实Resource GraphServing缺口；后者需要独立集成、冻结和逐次图重放证明。</p>
<p><a href="report.json">完整逐记录数据、原始路径和输入哈希</a></p>"""
    )
    (out / "index.html").write_text(page)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[3], args.out)
    print(
        json.dumps(
            {g: {k: v for k, v in c.items() if k != "records"} for g, c in result["cards"].items()},
            indent=2,
        )
    )
