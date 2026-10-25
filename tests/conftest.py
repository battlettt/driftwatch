import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


def make_frame(n: int, seed: int, shift: float = 0.0, cat_p=(0.5, 0.3, 0.2)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0 + shift, 1.0, n)
    x2 = rng.exponential(2.0, n)
    color = rng.choice(["red", "green", "blue"], size=n, p=list(cat_p))
    logit = 1.5 * x1 - 0.5 * x2 + 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "color": color, "y": y})


@pytest.fixture
def reference_df() -> pd.DataFrame:
    return make_frame(3000, seed=1)


@pytest.fixture
def stable_df() -> pd.DataFrame:
    return make_frame(2000, seed=2)


@pytest.fixture
def drifted_df() -> pd.DataFrame:
    return make_frame(2000, seed=3, shift=1.5, cat_p=(0.1, 0.3, 0.6))


@pytest.fixture
def model(reference_df):
    return LogisticRegression().fit(reference_df[["x1", "x2"]], reference_df["y"])
