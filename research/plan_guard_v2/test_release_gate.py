import copy
import math
import unittest

import numpy as np

from release_gate import evaluate_release_gate


def cell(selected, guarded=1.05, off=1.0, resolved=True):
    return {
        "calls": 16,
        "metric": "run_device_us",
        "selected": selected,
        "comparisons": {
            "guarded": {"ratio": guarded, "controls_resolve": resolved},
            "off": {"ratio": off, "controls_resolve": True},
        },
    }


def draws(cells):
    out = {}
    for i, c in enumerate(cells):
        for mode in ("guarded", "off"):
            out[(i, mode)] = np.full(2000, math.log(c["comparisons"][mode]["ratio"]))
    return out


NUMERICS = {
    mode: {"qualifications": 10, "exact_full_outputs": 10, "max_abs_vs_pristine": 0.0}
    for mode in ("pristine", "off", "cap", "guarded")
}


class ReleaseGateTests(unittest.TestCase):
    def test_passes_strong_selected_gain_and_identity_fallback(self):
        cells = [cell(True, 1.08), cell(False, 1.0)]
        result = evaluate_release_gate(cells, draws(cells), NUMERICS)
        self.assertTrue(result["pass"])
        self.assertEqual(result["selected"]["count"], 1)

    def test_selected_regression_fails(self):
        cells = [cell(True, 0.985), cell(False, 1.0)]
        result = evaluate_release_gate(cells, draws(cells), NUMERICS)
        self.assertFalse(result["pass"])
        self.assertFalse(result["requirements"]["selected_worst_point_at_least_0_99"])

    def test_unresolved_selected_cell_fails(self):
        cells = [cell(True, 1.08, resolved=False), cell(False, 1.0)]
        self.assertFalse(evaluate_release_gate(cells, draws(cells), NUMERICS)["pass"])

    def test_no_selected_cell_fails(self):
        cells = [cell(False, 1.0), cell(False, 1.0)]
        result = evaluate_release_gate(cells, draws(cells), NUMERICS)
        self.assertFalse(result["pass"])
        self.assertFalse(result["requirements"]["selected_nonempty"])

    def test_numerical_mismatch_fails(self):
        cells = [cell(True, 1.08), cell(False, 1.0)]
        numerics = copy.deepcopy(NUMERICS)
        numerics["guarded"]["exact_full_outputs"] = 9
        self.assertFalse(evaluate_release_gate(cells, draws(cells), numerics)["pass"])

    def test_policy_and_overlay_worst_cases_are_gated(self):
        cells = [cell(True, 1.08), cell(False, 0.98)]
        result = evaluate_release_gate(cells, draws(cells), NUMERICS)
        self.assertFalse(result["requirements"]["policy_worst_point_at_least_0_99"])
        cells = [cell(True, 1.08, off=0.98), cell(False, 1.0, off=1.0)]
        result = evaluate_release_gate(cells, draws(cells), NUMERICS)
        self.assertFalse(result["requirements"]["off_overlay_worst_point_at_least_0_99"])


if __name__ == "__main__":
    unittest.main()
