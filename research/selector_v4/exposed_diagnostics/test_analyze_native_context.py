"""Statistical checks for process-cluster uncertainty in diagnostic analysis."""

import numpy as np
import pytest

from research.selector_v4.exposed_diagnostics.analyze_native_context import interval


def test_constant_paired_gain_has_no_invented_uncertainty():
    values = np.full((3, 24), np.log(1.25))
    assert interval(values, np.random.default_rng(19), 0.95) == pytest.approx([1.25, 1.25])


def test_process_effect_is_retained_even_with_many_constant_blocks():
    # Each process has zero within-process variation but large process variation.
    # A bootstrap that treats all blocks as independent would hide this effect.
    values = np.repeat(np.log([1.0, 2.0, 4.0])[:, None], 24, axis=1)
    assert interval(values, np.random.default_rng(19), 0.95) == pytest.approx([1.0, 4.0])
