import numpy as np
import pytest
from scipy.special import expit
from sklearn.metrics import log_loss

from modeling.cnn.calibration import apply_temperature, fit_temperature


def test_compressed_probabilities_recover_temperature_without_boundary_collapse():
    # Exact calibration requires sigmoid(0.2/T)=0.9 => T=0.2/log(9).
    logits = np.repeat([-0.2, 0.2], 100)
    y = np.r_[np.zeros(90), np.ones(10), np.zeros(10), np.ones(90)]
    temperature = fit_temperature(logits, y)
    assert temperature == pytest.approx(0.2 / np.log(9), rel=1e-8)
    assert log_loss(y, apply_temperature(logits, temperature)) < log_loss(y, expit(logits))


def test_temperature_fit_is_never_worse_than_identity_on_fit_data():
    rng = np.random.default_rng(37)
    z = rng.normal(size=500)
    y = rng.binomial(1, expit(z))
    for scale in [0.01, 0.2, 1.0, 5.0, 100.0]:
        logits = z * scale
        t = fit_temperature(logits, y)

        def loss(beta, logits=logits):
            return np.mean(np.logaddexp(0.0, beta * logits) - y * beta * logits)

        assert 0.05 <= t <= 20.0
        assert loss(1 / t) <= loss(1.0) + 1e-12


def test_temperature_degenerate_and_nonfinite_inputs():
    assert fit_temperature(np.zeros(4), np.array([0, 1, 0, 1])) == 1.0
    with pytest.raises(ValueError):
        fit_temperature(np.array([np.nan, 1.0]), np.array([0, 1]))
