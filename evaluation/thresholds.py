from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd


def select_threshold_on_validation(
    validation: pd.DataFrame,
    thresholds: Iterable[float],
    evaluator: Callable[[pd.DataFrame, float], tuple[dict[str, float], pd.DataFrame]],
) -> tuple[float, pd.DataFrame]:
    summaries = [evaluator(validation, threshold)[0] for threshold in thresholds]
    summary = pd.DataFrame(summaries)
    sortable = summary.copy()
    sortable["profit_factor_sort"] = sortable["profit_factor"].replace(float("inf"), 1_000_000.0)
    best = sortable.sort_values(
        ["total_return", "max_drawdown", "profit_factor_sort"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["threshold"]), summary


def evaluate_fixed_threshold(
    test: pd.DataFrame,
    selected_threshold: float,
    evaluator: Callable[[pd.DataFrame, float], tuple[dict[str, float], pd.DataFrame]],
) -> tuple[dict[str, float], pd.DataFrame]:
    return evaluator(test, selected_threshold)
