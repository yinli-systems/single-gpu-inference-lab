import hashlib
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

from l20_stack.benchmark_protocol import (
    paired_trial_ratios,
    permuted_provider_order,
    summarize_paired_speedups,
    summarize_trials,
)


def load_top_logprobs_benchmark_module():
    path = Path("scripts/benchmark_l20_top_logprobs.py")
    spec = importlib.util.spec_from_file_location("benchmark_l20_top_logprobs", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trial_summary_keeps_per_trial_samples_and_median_of_medians():
    summary = summarize_trials(
        [
            [1.0, 3.0, 2.0],
            [4.0, 6.0, 5.0],
            [7.0, 9.0, 8.0],
        ]
    )

    assert summary["median_ms"] == 5.0
    assert summary["min_trial_median_ms"] == 2.0
    assert summary["max_trial_median_ms"] == 8.0
    assert summary["trial_medians_ms"] == [2.0, 5.0, 8.0]
    assert summary["trials"][0]["samples_ms"] == [1.0, 3.0, 2.0]


def test_paired_speedup_is_computed_within_each_trial():
    ratios = paired_trial_ratios(
        baseline_trial_medians=[8.0, 18.0, 28.0],
        candidate_trial_medians=[2.0, 3.0, 4.0],
    )
    summary = summarize_paired_speedups(ratios)

    assert ratios == [4.0, 6.0, 7.0]
    assert summary == {
        "median": 6.0,
        "min": 4.0,
        "max": 7.0,
        "all_trials_faster": True,
        "paired_trial_values": [4.0, 6.0, 7.0],
    }


def test_paired_speedup_rejects_unpaired_or_nonpositive_trials():
    with pytest.raises(ValueError, match="counts must match"):
        paired_trial_ratios([2.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="must be positive"):
        paired_trial_ratios([2.0], [0.0])


def test_provider_order_covers_all_permutations_and_balances_positions():
    providers = ("triton", "logsoftmax", "logsumexp")
    orders = [
        permuted_provider_order(
            providers,
            trial=2,
            round_index=round_index,
            seed=113,
        )
        for round_index in range(6)
    ]

    assert orders == [
        ("triton", "logsumexp", "logsoftmax"),
        ("logsoftmax", "triton", "logsumexp"),
        ("logsoftmax", "logsumexp", "triton"),
        ("logsumexp", "triton", "logsoftmax"),
        ("logsumexp", "logsoftmax", "triton"),
        ("triton", "logsoftmax", "logsumexp"),
    ]
    for provider in providers:
        assert [order.index(provider) for order in orders].count(0) == 2
        assert [order.index(provider) for order in orders].count(1) == 2
        assert [order.index(provider) for order in orders].count(2) == 2
    assert len(set(orders)) == 6


def test_top_logprobs_timing_contract_is_explicit_in_source():
    source = Path("scripts/benchmark_l20_top_logprobs.py").read_text(
        encoding="utf-8"
    )
    benchmark = source.split("def benchmark_interleaved(", 1)[1].split(
        "def make_clock_precondition(", 1
    )[0]
    measured_call = benchmark.split(
        "start, end = event_pairs[name][trial][round_index]", 1
    )[1]

    assert measured_call.index("precondition()") < measured_call.index("start.record()")
    assert measured_call.index("start.record()") < measured_call.index(
        "providers[name]()"
    )
    assert measured_call.index("providers[name]()") < measured_call.index("end.record()")
    assert benchmark.count("torch.cuda.synchronize()") == 2
    assert benchmark.rindex("torch.cuda.synchronize()") < benchmark.index(
        "start.elapsed_time(end)"
    )


def test_interleaved_protocol_executes_precondition_outside_every_event():
    module = load_top_logprobs_benchmark_module()
    log = []

    class FakeEvent:
        next_id = 0

        def __init__(self, *, enable_timing):
            assert enable_timing is True
            self.identifier = FakeEvent.next_id
            FakeEvent.next_id += 1
            log.append(("allocate", self.identifier))

        def record(self):
            log.append(("record", self.identifier))

        def elapsed_time(self, end):
            log.append(("elapsed", self.identifier, end.identifier))
            return 1.0

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def synchronize():
            log.append(("synchronize",))

    class FakeTorch:
        cuda = FakeCuda

    module.torch = FakeTorch
    providers = {
        name: (lambda provider=name: log.append(("provider", provider)))
        for name in ("triton", "logsoftmax", "logsumexp")
    }

    samples = module.benchmark_interleaved(
        providers,
        precondition=lambda: log.append(("precondition",)),
        warmup=0,
        rounds=6,
        trials=1,
        order_seed=113,
    )

    first_sync = log.index(("synchronize",))
    second_sync = log.index(("synchronize",), first_sync + 1)
    measured = log[first_sync + 1 : second_sync]
    assert len(measured) == 6 * 3 * 4
    for offset in range(0, len(measured), 4):
        precondition, start, provider, end = measured[offset : offset + 4]
        assert precondition == ("precondition",)
        assert start[0] == "record"
        assert provider[0] == "provider"
        assert end[0] == "record"
    assert all(entry[0] == "allocate" for entry in log[:first_sync])
    assert all(entry[0] == "elapsed" for entry in log[second_sync + 1 :])
    assert {name: len(trials[0]) for name, trials in samples.items()} == {
        "triton": 6,
        "logsoftmax": 6,
        "logsumexp": 6,
    }


def test_tie_aware_correctness_accepts_only_exact_cutoff_ties():
    torch = pytest.importorskip("torch")
    module = load_top_logprobs_benchmark_module()
    logits = torch.tensor([[5.0, 4.0, 3.0, 3.0, 2.999]], dtype=torch.float16)
    normalized = torch.log_softmax(logits.float(), dim=-1)

    reference_boundary = int(torch.topk(logits.float(), 3, dim=-1).indices[0, -1])
    alternate_boundary = 2 if reference_boundary == 3 else 3
    tied_tokens = torch.tensor([[0, 1, alternate_boundary]], dtype=torch.int64)
    tied_values = normalized.gather(-1, tied_tokens)
    valid = module.tie_aware_correctness(
        logits,
        tied_values,
        tied_tokens,
        temperature=1.0,
    )
    assert valid["tie_aware_match"] is True
    assert valid["tokens_match_exact_order"] is False

    reversed_tokens = torch.tensor([[3, 1, 0]], dtype=torch.int64)
    reversed_values = normalized.gather(-1, reversed_tokens)
    reversed_result = module.tie_aware_correctness(
        logits,
        reversed_values,
        reversed_tokens,
        temperature=1.0,
    )
    assert reversed_result["tie_aware_match"] is False
    assert reversed_result["output_is_nonincreasing"] is False

    below_cutoff_tokens = torch.tensor([[0, 1, 4]], dtype=torch.int64)
    below_cutoff_values = normalized.gather(-1, below_cutoff_tokens)
    below_cutoff = module.tie_aware_correctness(
        logits,
        below_cutoff_values,
        below_cutoff_tokens,
        temperature=1.0,
    )
    assert below_cutoff["tie_aware_match"] is False
    assert below_cutoff["selected_at_or_above_exact_cutoff"] is False

    missing_strict_tokens = torch.tensor([[0, 2, 3]], dtype=torch.int64)
    missing_strict_values = normalized.gather(-1, missing_strict_tokens)
    missing_strict = module.tie_aware_correctness(
        logits,
        missing_strict_values,
        missing_strict_tokens,
        temperature=1.0,
    )
    assert missing_strict["tie_aware_match"] is False
    assert missing_strict["all_strict_top_tokens_present"] is False


def test_top_logprobs_schema_records_provenance_and_clock_policy():
    source = Path("scripts/benchmark_l20_top_logprobs.py").read_text(
        encoding="utf-8"
    )

    for marker in (
        '"schema_version": 2',
        '"benchmark_script_sha256"',
        '"kernel_source_sha256"',
        '"generated_at_utc"',
        '"torch_cuda_runtime"',
        '"compute_capability"',
        '"nvidia_smi_snapshots"',
        '"steady-state-gemm"',
        '"included_in_timing": False',
        '"deterministic_all_permutations_for_within_round_position_balance"',
        '"paired_speedups"',
        "allow_nan=False",
        '"timing_excludes": "host launch and allocator latency"',
    ):
        assert marker in source


def test_checked_in_top_logprobs_revalidation_matches_raw_trials():
    artifact_root = Path("benchmarks/results/a100-fused-top-logprobs")
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == 2
    assert summary["evidence_status"] == "controlled_revalidation"
    assert summary["collection"]["independent_processes_per_shape"] == 3
    assert summary["collection"]["distinct_seeds_per_shape"] == [113, 114, 115]
    assert summary["collection"]["total_paired_trials_per_shape"] == 15
    assert summary["provenance"]["all_runs_clean"] is True
    assert summary["timing_protocol"]["clock_policy"] == "steady-state-gemm"
    assert summary["timing_protocol"]["precondition"] == {
        "operation": "torch.mm",
        "shape": [8192, 8192],
        "dtype": "float16",
        "repeats_per_provider_sample": 1,
        "same_cuda_stream": True,
        "included_in_cuda_event_interval": False,
    }

    script_hash = hashlib.sha256(
        Path("scripts/benchmark_l20_top_logprobs.py").read_bytes()
    ).hexdigest()
    assert summary["provenance"]["benchmark_script_sha256"] == script_hash

    # The kernel source may move on after a measurement. Every such change must
    # be acknowledged in the artifact as a dated supersession entry that names
    # the new source hash and states whether performance was remeasured, so the
    # A100 numbers are never silently re-attributed to code they did not run.
    kernel_hash = hashlib.sha256(
        Path("src/l20_stack/ops/triton_sampling.py").read_bytes()
    ).hexdigest()
    acknowledged = {summary["provenance"]["kernel_source_sha256"]}
    for entry in summary["provenance"].get("kernel_source_supersessions", []):
        assert entry["date"] and entry["change"] and entry["performance_effect"]
        acknowledged.add(entry["kernel_source_sha256"])
    assert kernel_hash in acknowledged, (
        "triton_sampling.py changed since the artifact was measured; add a "
        "kernel_source_supersessions entry to summary.json describing the change"
    )

    for row in summary["rows"]:
        payloads = [
            json.loads((artifact_root / relative).read_text(encoding="utf-8"))
            for relative in row["raw_files"]
        ]
        assert len(payloads) == 3
        assert {
            payload["timing_policy"]["provider_order"]["seed"]
            for payload in payloads
        } == {113, 114, 115}
        for payload in payloads:
            assert payload["schema_version"] == 2
            assert payload["shape"]["batch"] == row["batch"]
            assert payload["provenance"]["commit"] == summary["provenance"]["repo_commit"]
            assert payload["provenance"]["dirty"] is False
            assert payload["timing_policy"]["trials"] == 5
            assert (
                payload["timing_policy"]["measured_rounds_per_provider_per_trial"]
                == 30
            )
            assert payload["timing_policy"]["clock"]["policy"] == "steady-state-gemm"
            assert payload["correctness"]["tie_aware_match"] is True
            assert payload["shape"] == {
                "batch": row["batch"],
                "dtype": "float16",
                "temperature": 0.8,
                "top_n": 5,
                "vocab": 151936,
            }
            assert payload["environment"]["python"] == summary["software"]["python"]
            assert payload["environment"]["torch"] == summary["software"]["torch"]
            assert (
                payload["environment"]["torch_cuda_runtime"]
                == summary["software"]["torch_cuda_runtime"]
            )
            assert payload["environment"]["triton"] == summary["software"]["triton"]
            assert payload["environment"]["gpu"]["compute_capability"] == [8, 0]
            assert (
                payload["provenance"]["benchmark_script_sha256"]
                == summary["provenance"]["benchmark_script_sha256"]
            )
            assert (
                payload["provenance"]["kernel_source_sha256"]
                == summary["provenance"]["kernel_source_sha256"]
            )
            assert payload["provenance"]["publication_normalization"] == (
                "GPU UUID hashed; executable and output paths normalized"
            )
            serialized = json.dumps(payload)
            assert "/root/" not in serialized
            assert "/tmp/codex-toplogprobs" not in serialized
            assert "a977cc53-eb3f-6730-ca10-7e9ac7e40ba1" not in serialized

        provider_pairs = (
            (
                "triton_top_logprobs_preallocated",
                "triton_top_logprobs_preallocated",
            ),
            ("torch_logsoftmax_then_topk", "torch_logsoftmax_then_topk"),
            ("torch_logsumexp_then_topk", "torch_logsumexp_then_topk"),
        )
        for summary_key, raw_key in provider_pairs:
            process_medians = [
                payload["providers"][raw_key]["median_ms"] for payload in payloads
            ]
            aggregate = row[summary_key]
            assert aggregate["median_of_process_medians_ms"] == pytest.approx(
                statistics.median(process_medians)
            )
            assert aggregate["min_process_median_ms"] == pytest.approx(
                min(process_medians)
            )
            assert aggregate["max_process_median_ms"] == pytest.approx(
                max(process_medians)
            )

        for summary_key, raw_key in (
            (
                "paired_speedup_vs_torch_logsoftmax_then_topk",
                "vs_torch_logsoftmax_then_topk",
            ),
            (
                "paired_speedup_vs_torch_logsumexp_then_topk",
                "vs_torch_logsumexp_then_topk",
            ),
        ):
            paired_trials = [
                value
                for payload in payloads
                for value in payload["paired_speedups"][raw_key][
                    "paired_trial_values"
                ]
            ]
            aggregate = row[summary_key]
            assert len(paired_trials) == 15
            assert aggregate["median"] == pytest.approx(
                statistics.median(paired_trials)
            )
            assert aggregate["min"] == pytest.approx(min(paired_trials))
            assert aggregate["max"] == pytest.approx(max(paired_trials))
            assert aggregate["all_15_trials_faster"] is all(
                value > 1.0 for value in paired_trials
            )

    for historical in summary["historical_artifact"]["files"]:
        actual_hash = hashlib.sha256(
            (artifact_root / historical["path"]).read_bytes()
        ).hexdigest()
        assert historical["sha256"] == actual_hash

    public_claims = {
        Path("README.md"): "8.39x–9.45x",
        Path("docs/reviewer-guide.md"): "8.39x–9.45x",
        Path("docs/experiment-status.md"): "8.39x-9.45x",
        Path("benchmarks/results/README.md"): "8.39x-9.45x",
        artifact_root / "README.md": "8.39x–9.45x",
    }
    for path, claim in public_claims.items():
        assert claim in path.read_text(encoding="utf-8")
    assert summary["claim"]["paired_speedup_range"] == "8.39x-9.45x"


def test_checked_in_l20_post_fix_top_logprobs_artifact_matches_raw_trials():
    artifact_root = Path("benchmarks/results/l20-fused-top-logprobs-2026-09")
    summary = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == 2
    assert summary["evidence_status"] == "controlled_revalidation"
    assert summary["hardware"]["gpu"] == "NVIDIA L20"
    assert summary["hardware"]["compute_capability"] == [8, 9]
    assert summary["collection"]["independent_processes_per_shape"] == 3
    assert summary["collection"]["distinct_seeds_per_shape"] == [113, 114, 115]
    assert summary["collection"]["total_paired_trials_per_shape"] == 15
    assert summary["provenance"]["all_runs_clean"] is True
    assert summary["timing_protocol"]["clock_policy"] == "steady-state-gemm"
    assert summary["shape"] == {
        "vocab": 151_936,
        "top_n": 5,
        "temperature": 0.8,
        "dtype": "float16",
    }

    # This artifact is the post-fix measurement: its kernel hash must be the
    # one the A100 artifact's supersession entry points at.
    a100 = json.loads(
        Path("benchmarks/results/a100-fused-top-logprobs/summary.json").read_text(
            encoding="utf-8"
        )
    )
    superseding = {
        entry["kernel_source_sha256"]
        for entry in a100["provenance"]["kernel_source_supersessions"]
    }
    assert summary["provenance"]["kernel_source_sha256"] in superseding

    for relative, expected_hash in summary["provenance"]["raw_file_sha256"].items():
        assert hashlib.sha256((artifact_root / "raw" / relative).read_bytes()).hexdigest() == (
            expected_hash
        )

    for row in summary["rows"]:
        payloads = [
            json.loads((artifact_root / relative).read_text(encoding="utf-8"))
            for relative in row["raw_files"]
        ]
        assert len(payloads) == 3
        assert {
            payload["timing_policy"]["provider_order"]["seed"] for payload in payloads
        } == {113, 114, 115}
        for payload in payloads:
            assert payload["schema_version"] == 2
            assert payload["shape"]["batch"] == row["batch"]
            assert payload["provenance"]["commit"] == summary["provenance"]["repo_commit"]
            assert payload["provenance"]["dirty"] is False
            assert payload["environment"]["gpu"]["name"] == "NVIDIA L20"
            assert payload["correctness"]["tie_aware_match"] is True
            assert payload["correctness"]["max_abs_logprob_error"] <= 5e-7
        for baseline in ("torch_logsoftmax_then_topk", "torch_logsumexp_then_topk"):
            paired = [
                value
                for payload in payloads
                for value in payload["paired_speedups"][f"vs_{baseline}"]["paired_trial_values"]
            ]
            assert len(paired) == 15
            recorded = row[f"paired_speedup_vs_{baseline}"]
            assert recorded["median"] == statistics.median(paired)
            assert recorded["min"] == min(paired)
            assert recorded["max"] == max(paired)
            assert recorded["all_15_trials_faster"] is True
            assert min(paired) > 1.0


def test_sampling_mask_artifacts_are_internally_consistent():
    """The serving A/B summary must hash its raw files and reproduce its headline rows."""
    root = Path("benchmarks/results/l20-vllm-sampling-mask-ab")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for run in summary["runs"].values():
        raw = root / "raw" / run["raw_file"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == run["raw_sha256"]
        assert run["provenance"]["dirty"] is False
        rows = {(r["server"], r["request"]): r for r in run["rows"]}
        for row in rows.values():
            assert row["tok_per_s_min"] <= row["tok_per_s_median"] <= row["tok_per_s_max"]
    main = summary["runs"]["qwen25-05b-generate"]
    rows = {(r["server"], r["request"]): r for r in main["rows"]}
    assert rows[("mask_bitmap", "gen")]["tok_per_s_vs_native_gen"] < 0.1
    assert rows[("mask_compact", "gen")]["tok_per_s_vs_native_gen"] > 0.7
    big = {(r["server"], r["request"]): r for r in summary["runs"]["qwen3-4b-generate"]["rows"]}
    assert big[("mask_compact", "gen")]["tok_per_s_vs_native_gen"] > 0.9
    # equivalence: identical tokens under batch-invariant mode, and mask agreement no
    # worse than the bitmap server's own run-to-run agreement
    eq = summary["equivalence"]
    for key in ("batch_invariant-gen", "batch_invariant-logprobs"):
        assert eq[key]["token_sequences_identical"] == eq[key]["requests"]
    control = eq["batch_invariant-bitmap-run-a-vs-b"]
    assert control["token_sequences_identical"] == control["requests"]
    assert eq["batch_invariant-gen"]["masks_identical"] >= control["masks_identical"]

    path = json.loads(Path("benchmarks/results/l20-support-pack-path/raw.json").read_text(encoding="utf-8"))
    assert path["provenance"]["dirty"] is False
    assert path["provenance"]["upstream_output_py_sha256"] == (
        "1a1abac89cc6cc2278b6f984e4e20dbf82df81130112a875eeb9d72a694f8230"
    )
    for row in path["rows"]:
        assert row["speedup_total"] > 30
        assert row["bytes_d2h"]["compact"] < row["bytes_d2h"]["upstream"] / 50
