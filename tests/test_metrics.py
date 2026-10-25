import math

import numpy as np
import pytest

from driftwatch.metrics import (
    bin_counts,
    chi_square_test,
    js_divergence,
    ks_test,
    psi,
    quantile_edges,
    total_variation,
)

# --------------------------------------------------------------------- PSI


def test_psi_identical_distributions_is_zero():
    p = [0.1, 0.2, 0.3, 0.4]
    assert psi(p, p) == pytest.approx(0.0, abs=1e-12)


def test_psi_accepts_counts_or_proportions():
    assert psi([10, 20, 70], [20, 20, 60]) == pytest.approx(psi([0.1, 0.2, 0.7], [0.2, 0.2, 0.6]))


def test_psi_empty_bin_in_reference_is_finite():
    value = psi([0.5, 0.5, 0.0], [0.4, 0.4, 0.2])
    assert math.isfinite(value)
    assert value > 0.25  # mass appearing in a never-seen bin is a big shift


def test_psi_empty_bin_in_current_is_finite():
    value = psi([0.4, 0.4, 0.2], [0.5, 0.5, 0.0])
    assert math.isfinite(value)
    assert value > 0


def test_psi_both_empty_bin_contributes_nothing():
    assert psi([0.3, 0.7, 0.0], [0.3, 0.7, 0.0]) == pytest.approx(0.0, abs=1e-12)
    assert psi([0.3, 0.7, 0.0], [0.4, 0.6, 0.0]) == pytest.approx(psi([0.3, 0.7], [0.4, 0.6]), rel=1e-3)


def test_psi_known_value():
    e, a = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    expected = float(np.sum((a - e) * np.log(a / e)))
    assert psi(e, a) == pytest.approx(expected)


def test_psi_is_non_negative_and_grows_with_shift():
    ref = [0.25, 0.25, 0.25, 0.25]
    small = psi(ref, [0.2, 0.25, 0.25, 0.3])
    large = psi(ref, [0.05, 0.15, 0.3, 0.5])
    assert 0 < small < large


@pytest.mark.parametrize(
    "expected,actual",
    [([], []), ([0, 0], [0.5, 0.5]), ([0.5, 0.5], [1.0]), ([-1, 2], [0.5, 0.5]), ([np.nan, 1], [0.5, 0.5])],
)
def test_psi_rejects_invalid_input(expected, actual):
    with pytest.raises(ValueError):
        psi(expected, actual)


# --------------------------------------------------------------- JS / TV


def test_js_divergence_bounds():
    assert js_divergence([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    assert 0 < js_divergence([0.5, 0.5], [0.4, 0.6]) < 1


def test_js_divergence_is_symmetric():
    p, q = [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))


def test_total_variation():
    assert total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert total_variation([1, 0], [0, 1]) == pytest.approx(1.0)
    assert total_variation([0.5, 0.3, 0.2], [0.3, 0.3, 0.4]) == pytest.approx(0.2)


# -------------------------------------------------------------------- KS


def test_ks_identical_data_is_not_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    stat, p = ks_test(x, x)
    assert stat == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_ks_same_distribution_not_significant():
    rng = np.random.default_rng(1)
    stat, p = ks_test(rng.normal(size=2000), rng.normal(size=2000))
    assert stat < 0.05
    assert p > 0.01


def test_ks_shifted_data_is_significant():
    rng = np.random.default_rng(2)
    stat, p = ks_test(rng.normal(size=2000), rng.normal(0.5, 1.0, size=2000))
    assert stat > 0.15
    assert p < 1e-6


def test_ks_ignores_nans_and_rejects_empty():
    stat, _ = ks_test([1.0, 2.0, np.nan], [1.0, 2.0])
    assert stat == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ks_test([np.nan], [1.0])


# ------------------------------------------------------------ chi-square


def test_chi_square_same_counts_not_significant():
    stat, p = chi_square_test([500, 300, 200], [250, 150, 100])
    assert stat == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_chi_square_shifted_counts_significant():
    _, p = chi_square_test([500, 300, 200], [100, 300, 600])
    assert p < 1e-10


def test_chi_square_handles_new_and_empty_categories():
    # third bucket unseen in reference, fourth empty in both
    stat, p = chi_square_test([600, 400, 0, 0], [300, 200, 100, 0])
    assert math.isfinite(stat) and p < 1e-6


def test_chi_square_single_bucket():
    assert chi_square_test([10, 0], [5, 0]) == (0.0, 1.0)


# --------------------------------------------------------------- binning


def test_quantile_edges_dedupes_ties():
    values = [0] * 80 + [1] * 15 + [2] * 5
    edges = quantile_edges(values, 10)
    assert edges == sorted(set(edges))
    assert len(edges) <= 3


def test_bin_counts_open_ended_with_missing_bin():
    counts = bin_counts([-100, 0.5, 1.0, 1.5, 999, np.nan], [1.0])
    # bins: (-inf, 1), [1, inf), missing
    assert counts.tolist() == [2, 3, 1]
