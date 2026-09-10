from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from create_risk_targets import add_risk_targets
from create_similarity_dataset import select_and_scale_features
from evaluation.splits import purged_chronological_split
from evaluation.targets import apply_threshold, fit_training_quantile_threshold
from evaluation.thresholds import evaluate_fixed_threshold, select_threshold_on_validation


class LeakageControlTests(unittest.TestCase):
    def test_scaler_is_unchanged_by_future_append(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
        past = pd.DataFrame({"timestamp": timestamps, "return_1": np.arange(8, dtype=float)})
        extended = pd.concat(
            [past, pd.DataFrame({"timestamp": [timestamps[-1] + pd.Timedelta(minutes=15)], "return_1": [1e9]})],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            scaler_path = Path(directory) / "scaler.csv"
            scaled_a, _, _ = select_and_scale_features(past, scaler_path=scaler_path)
            metadata_a = pd.read_csv(scaler_path)
            scaled_b, _, _ = select_and_scale_features(extended, scaler_path=scaler_path)
            metadata_b = pd.read_csv(scaler_path)
        pd.testing.assert_frame_equal(metadata_a, metadata_b)
        pd.testing.assert_frame_equal(scaled_a, scaled_b.iloc[: len(past)].reset_index(drop=True))

    def test_target_threshold_is_unchanged_by_future_append(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
        raw = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        fit_end = timestamps[4]
        first = fit_training_quantile_threshold(
            raw, pd.Series(timestamps), fit_end, percentile=0.70, horizon=4,
            target_type="future_volatility",
        )
        second = fit_training_quantile_threshold(
            pd.concat([raw, pd.Series([1e9])], ignore_index=True),
            pd.Series(timestamps.append(pd.DatetimeIndex([timestamps[-1] + pd.Timedelta(minutes=15)]))),
            fit_end, percentile=0.70, horizon=4, target_type="future_volatility",
        )
        self.assertEqual(first.value, second.value)

    def test_historical_labels_are_unchanged_by_future_append(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=30, freq="15min", tz="UTC")
        close = 100 + np.sin(np.arange(30)) + np.arange(30) * 0.1
        past = pd.DataFrame({"timestamp": timestamps, "close": close})
        fit_end = timestamps[20]
        labeled_a, threshold_a = add_risk_targets(past, threshold_fit_end=fit_end)
        future = pd.DataFrame(
            {"timestamp": pd.date_range(timestamps[-1] + pd.Timedelta(minutes=15), periods=5, freq="15min", tz="UTC"),
             "close": [1000, 10, 2000, 5, 3000]}
        )
        labeled_b, threshold_b = add_risk_targets(pd.concat([past, future], ignore_index=True), threshold_fit_end=fit_end)
        self.assertEqual(threshold_a, threshold_b)
        pd.testing.assert_series_equal(
            labeled_a.loc[:20, "target_volatility_high_next_4"].reset_index(drop=True),
            labeled_b.loc[:20, "target_volatility_high_next_4"].reset_index(drop=True),
        )

    def test_boundary_rows_are_purged(self) -> None:
        timestamps = pd.date_range("2025-01-01", periods=40, freq="15min", tz="UTC")
        frame = pd.DataFrame({"timestamp": timestamps, "value": range(40)})
        train, validation, _, metadata = purged_chronological_split(frame, 4)
        self.assertEqual(metadata.train_rows_before_purge - len(train), 4)
        self.assertLess(timestamps[train.index[-1]], metadata.validation_start)
        self.assertTrue((validation["timestamp"] < metadata.test_start).all())

    def test_irregular_timestamps_use_label_end_time(self) -> None:
        timestamps = pd.to_datetime(
            ["2025-01-01 00:00", "2025-01-01 00:15", "2025-01-01 00:30", "2025-01-01 00:45",
             "2025-01-01 03:00", "2025-01-01 03:15", "2025-01-01 03:30", "2025-01-01 03:45",
             "2025-01-01 04:00", "2025-01-01 04:15"], utc=True
        )
        frame = pd.DataFrame({"timestamp": timestamps, "label_end": timestamps})
        frame.loc[4, "label_end"] = timestamps[8]
        train, _, _, metadata = purged_chronological_split(
            frame, 1, label_end_column="label_end", train_fraction=0.60, validation_fraction=0.20
        )
        self.assertNotIn(timestamps[4], set(train["timestamp"]))
        self.assertTrue((pd.to_datetime(train["label_end"], utc=True) < metadata.validation_start).all())

    def test_test_data_cannot_reselect_validation_threshold(self) -> None:
        validation = pd.DataFrame({"split": ["validation"]})
        test = pd.DataFrame({"split": ["test"]})

        def evaluator(frame: pd.DataFrame, threshold: float):
            preferred = 0.60 if frame.iloc[0]["split"] == "validation" else 0.80
            score = 1.0 if threshold == preferred else 0.0
            return {"threshold": threshold, "total_return": score, "max_drawdown": 0.0,
                    "profit_factor": 1.0}, frame.copy()

        selected, _ = select_threshold_on_validation(validation, [0.60, 0.80], evaluator)
        final, _ = evaluate_fixed_threshold(test, selected, evaluator)
        self.assertEqual(selected, 0.60)
        self.assertEqual(final["threshold"], 0.60)
        self.assertEqual(final["total_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
