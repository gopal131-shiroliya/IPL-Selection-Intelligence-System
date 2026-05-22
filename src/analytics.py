from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series:
    numerator_series = numerator if isinstance(numerator, pd.Series) else pd.Series([numerator])
    denominator_series = denominator if isinstance(denominator, pd.Series) else pd.Series([denominator])
    result = numerator_series.divide(denominator_series.replace(0, np.nan)).fillna(0.0)
    return result


def phase_for_over(over_number: int) -> str:
    if over_number < 6:
        return "powerplay"
    if over_number < 15:
        return "middle"
    return "death"
