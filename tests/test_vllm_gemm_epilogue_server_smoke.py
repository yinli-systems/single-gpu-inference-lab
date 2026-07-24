import importlib.util
import json
from pathlib import Path


def load_smoke():
    path = Path("scripts/smoke_vllm_l20_gemm_epilogue_server.py")
    spec = importlib.util.spec_from_file_location("smoke_vllm_l20_gemm_epilogue_server", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gemm_epilogue_server_smoke_summarizes_good_trace(tmp_path):
    module = load_smoke()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "eligible": True,
                "metadata": {
                    "mutates_outputs": True,
                    "api": {"api_called": True},
                    "epilogue": {
                        "returned_output": True,
                        "uses_full_logits": False,
                        "fallback_to_compute_logits": False,
                        "correctness": {
                            "checked": True,
                            "matches_baseline_argmax": True,
                            "expected_tokens": [11],
                            "actual_tokens": [11],
                        },
                    },
                },
                "reasons": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_trace(trace, expected_events=1)

    assert summary["trace_events"] == 1
    assert summary["expected_trace_events"] == 1
    assert summary["trace_count_matches_completion"] is True
    assert summary["good_gemm_epilogue_events"] == 1
    assert summary["all_gemm_epilogue_events_ok"] is True
    assert summary["token_pairs"] == [{"expected": [11], "actual": [11]}]


def test_gemm_epilogue_server_smoke_rejects_fallback_trace(tmp_path):
    module = load_smoke()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "eligible": True,
                "metadata": {
                    "mutates_outputs": False,
                    "api": {"api_called": True},
                    "epilogue": {
                        "returned_output": False,
                        "uses_full_logits": True,
                        "fallback_to_compute_logits": True,
                        "correctness": {
                            "checked": False,
                            "matches_baseline_argmax": False,
                        },
                    },
                },
                "reasons": ["fallback"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_trace(trace, expected_events=1)

    assert summary["trace_events"] == 1
    assert summary["good_gemm_epilogue_events"] == 0
    assert summary["all_gemm_epilogue_events_ok"] is False
    assert summary["bad_events"][0]["reasons"] == ["fallback"]


def test_gemm_epilogue_server_smoke_rejects_truncated_trace(tmp_path):
    module = load_smoke()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "eligible": True,
                "metadata": {
                    "mutates_outputs": True,
                    "api": {"api_called": True},
                    "epilogue": {
                        "returned_output": True,
                        "uses_full_logits": False,
                        "fallback_to_compute_logits": False,
                        "correctness": {
                            "checked": True,
                            "matches_baseline_argmax": True,
                        },
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = module.summarize_trace(trace, expected_events=8)

    assert summary["trace_count_matches_completion"] is False
    assert summary["all_gemm_epilogue_events_ok"] is False


def test_gemm_epilogue_server_smoke_sets_required_env():
    source = Path("scripts/smoke_vllm_l20_gemm_epilogue_server.py").read_text()
    assert "VLLM_L20_GEMM_EPILOGUE_ENABLE" in source
    assert "VLLM_USE_FLASHINFER_SAMPLER" in source
    assert "HF_HUB_OFFLINE" in source
    assert "/v1/completions" in source
    assert "all_gemm_epilogue_events_ok" in source
    assert "start_new_session=True" in source


def test_checked_in_gemm_epilogue_smoke_covers_every_completion_token():
    artifact = json.loads(
        Path(
            "benchmarks/results/l20-vllm-gemm-epilogue-server-smoke/summary.json"
        ).read_text(encoding="utf-8")
    )

    assert artifact["completion"]["completion_tokens"] == 8
    assert artifact["trace"]["events"] == 8
    assert artifact["trace"]["expected_trace_events"] == 8
    assert artifact["trace"]["trace_count_matches_completion"] is True
    assert artifact["trace"]["all_gemm_epilogue_events_ok"] is True


def test_gemm_epilogue_server_smoke_terminates_process_group(monkeypatch):
    module = load_smoke()
    signals = []

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout):
            assert timeout == 15
            return 0

    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )
    monkeypatch.setattr(module, "_process_group_exists", lambda process_group_id: False)

    module.terminate(Process())

    assert signals == [(123, module.signal.SIGTERM)]


def test_gemm_epilogue_server_smoke_cleans_group_after_parent_exit(monkeypatch):
    module = load_smoke()
    signals = []

    class Process:
        pid = 456

        def poll(self):
            return 0

    group_states = iter((True, False, False))
    monkeypatch.setattr(
        module.os,
        "killpg",
        lambda pid, sent_signal: signals.append((pid, sent_signal)),
    )
    monkeypatch.setattr(
        module,
        "_process_group_exists",
        lambda process_group_id: next(group_states),
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)

    module.terminate(Process())

    assert signals == [(456, module.signal.SIGTERM)]
