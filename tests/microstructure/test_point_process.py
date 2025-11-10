import numpy as np

from src.microstructure.point_process import HawkesProcess, PoissonProcess


def test_poisson_fit_and_sample():
    rng = np.random.default_rng(1)
    events = np.cumsum(rng.exponential(0.3, size=100))
    model = PoissonProcess().fit(events)
    assert model.rate > 0

    sampled = model.sample(horizon=1.0, rng=np.random.default_rng(2))
    assert np.all(sampled[:-1] < sampled[1:])
    assert np.all(sampled >= 0)


def test_hawkes_intensity_positive():
    rng = np.random.default_rng(7)
    base_events = np.cumsum(rng.exponential(0.2, size=120))
    model = HawkesProcess().fit(base_events)
    times = np.linspace(base_events[0], base_events[-1], num=10)
    intensities = model.intensity(times)
    assert np.all(intensities > 0)
    samples = model.sample(horizon=0.5, rng=np.random.default_rng(3))
    assert np.all(np.diff(samples) > 0) if len(samples) > 1 else True
