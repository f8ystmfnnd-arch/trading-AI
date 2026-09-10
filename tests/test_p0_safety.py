from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import create_similarity_dataset as similarity_dataset
from market.instrument import BYBIT_CATEGORY, BYBIT_REST_TICKER_URL, BYBIT_WS_URL
from risk.policy import (
    ACTION_SEVERITY,
    CAUTION,
    NO_TRADE,
    NORMAL,
    STALE,
    UNKNOWN,
    action_from_levels,
    action_from_probabilities,
    decide_action,
)


class SimilarityDatasetSafetyTests(unittest.TestCase):
    def test_repeated_generation_replaces_instead_of_appending(self) -> None:
        source = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC")}
        )
        scaled = pd.DataFrame({"return_1": [1.0, 2.0, 3.0, 4.0]})

        with tempfile.TemporaryDirectory() as directory:
            original_output_dir = similarity_dataset.OUTPUT_DIR
            similarity_dataset.OUTPUT_DIR = Path(directory)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    first = similarity_dataset.process_length(scaled, source, ["return_1"], [], 2)
                    second = similarity_dataset.process_length(scaled, source, ["return_1"], [], 2)
            finally:
                similarity_dataset.OUTPUT_DIR = original_output_dir

            raw_output = Path(second["raw_path"])
            summary_output = Path(second["summary_path"])
            raw_result = pd.read_csv(raw_output)
            summary_result = pd.read_csv(summary_output)
            raw_header_count = sum(
                line.startswith("window_start_timestamp,")
                for line in raw_output.read_text(encoding="utf-8").splitlines()
            )

        self.assertEqual(first["raw_shape"], (3, 5))
        self.assertEqual(second["raw_shape"], first["raw_shape"])
        self.assertEqual(len(raw_result), 3)
        self.assertEqual(len(summary_result), 3)
        self.assertEqual(raw_header_count, 1)
        self.assertTrue(pd.to_datetime(raw_result["window_end_timestamp"], utc=True).is_monotonic_increasing)


class RiskPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)
        self.metadata = {
            "feature_asof": self.now - timedelta(minutes=15),
            "predicted_at": self.now - timedelta(minutes=14),
            "valid_until": self.now + timedelta(minutes=1),
            "now": self.now,
        }

    def test_missing_predictions_are_not_normal(self) -> None:
        self.assertEqual(decide_action(None, None), UNKNOWN)

    def test_stale_prediction_is_not_normal(self) -> None:
        stale = dict(self.metadata)
        stale["valid_until"] = self.now - timedelta(seconds=1)
        self.assertEqual(decide_action(0.1, 0.01, **stale), STALE)

    def test_timestamp_mismatch_is_not_normal(self) -> None:
        self.assertEqual(
            decide_action(0.1, 0.01, timestamps_match=False, **self.metadata),
            UNKNOWN,
        )

    def test_drop_risk_is_monotonic(self) -> None:
        states = [
            action_from_probabilities(0.1, 0.01),
            action_from_probabilities(0.1, 0.08),
            action_from_probabilities(0.1, 0.16),
        ]
        self.assertEqual(states, [NORMAL, CAUTION, NO_TRADE])
        self.assertEqual(states, sorted(states, key=ACTION_SEVERITY.get))
        self.assertEqual(action_from_levels("LOW", "HIGH"), NO_TRADE)


class InstrumentConsistencyTests(unittest.TestCase):
    def test_historical_and_live_market_are_spot(self) -> None:
        self.assertEqual(BYBIT_CATEGORY, "spot")
        self.assertIn("category=spot", BYBIT_REST_TICKER_URL)
        self.assertTrue(BYBIT_WS_URL.endswith("/spot"))
        project_root = Path(__file__).resolve().parents[1]
        collector_source = (project_root / "collect_bybit_1m.py").read_text(encoding="utf-8-sig")
        dashboard_source = (project_root / "dashboard" / "app.py").read_text(encoding="utf-8-sig")
        similarity_source = (project_root / "analyze_similar_patterns.py").read_text(encoding="utf-8-sig")
        self.assertIn("from market.instrument import BYBIT_CATEGORY, BYBIT_SYMBOL", collector_source)
        self.assertIn("BYBIT_REST_TICKER_URL", dashboard_source)
        self.assertNotIn("category=linear", dashboard_source)
        self.assertIn("from risk.policy import", similarity_source)


if __name__ == "__main__":
    unittest.main()
