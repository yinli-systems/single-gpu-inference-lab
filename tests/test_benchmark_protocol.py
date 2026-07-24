import importlib.util
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
