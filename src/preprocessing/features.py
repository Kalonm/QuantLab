"""Feature engineering utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def polynomial_features(
    X: np.ndarray,
    degree: int = 2,
    *,
    include_bias: bool = False,
    interaction_only: bool = False,
    order: str = "C",
    feature_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, Optional[list[str]]]:
    """Generate polynomial feature expansions.

    Parameters
    ----------
    X:
        Input feature matrix.
    degree:
        Maximum polynomial degree.
    include_bias, interaction_only, order:
        Passed through to :class:`~sklearn.preprocessing.PolynomialFeatures`.
    feature_names:
        Optional list of feature names; if provided, expanded feature names will be
        returned using ``PolynomialFeatures.get_feature_names_out``.
    """

    transformer = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only,
        order=order,
    )
    transformed = transformer.fit_transform(X)
    if feature_names is not None:
        names = transformer.get_feature_names_out(feature_names).tolist()
    else:
        names = None
    return transformed, names


__all__ = ["polynomial_features"]
