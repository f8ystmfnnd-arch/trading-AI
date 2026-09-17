import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from refresh_live_data import COLUMNS, TIMEFRAMES, atomic_append_minutes, atomic_csv, build_live_features, complete_resample, validate_recent_minutes


def fixture(minutes=18000):
    timestamps = pd.date_range("2026-01-01", periods=minutes, freq="min", tz="UTC")
    close = 100 + np.sin(np.arange(minutes) / 13) + np.arange(minutes) / 10000
    return pd.DataFrame({"timestamp": timestamps, "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": 10.0, "turnover": 1000.0})[COLUMNS]


class LivePipelineTests(unittest.TestCase):
    def test_archive_append_has_one_header_and_rejects_duplicate_rows(self):
        raw = fixture(4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.csv"
            atomic_csv(raw.iloc[:2], path)
            atomic_append_minutes(raw.iloc[2:], path)
            self.assertEqual(len(pd.read_csv(path)), 4)
            self.assertEqual(path.read_text().count("timestamp,"), 1)
            with self.assertRaises(ValueError):
                atomic_append_minutes(raw.iloc[2:], path)
            self.assertEqual(len(pd.read_csv(path)), 4)

    def test_atomic_replace_retries_a_temporary_windows_lock(self):
        original_replace = Path.replace
        attempts = []
        def locked_once(source, target):
            attempts.append(target)
            if len(attempts) == 1:
                raise PermissionError("reader lock")
            return original_replace(source, target)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.csv"
            with patch.object(Path, "replace", locked_once), patch("refresh_live_data.time.sleep"):
                atomic_csv(pd.DataFrame({"value": [42]}), path)
            self.assertEqual(pd.read_csv(path).value.iloc[0], 42)
            self.assertEqual(len(attempts), 2)

    def test_incomplete_candles_are_excluded(self):
        raw = fixture(31)
        cutoff = raw.timestamp.iloc[-1] + pd.Timedelta(minutes=1)
        candles = complete_resample(raw, 15, cutoff)
        self.assertEqual(len(candles), 2)
        gapped = raw.drop(index=10)
        self.assertEqual(len(complete_resample(gapped, 15, cutoff)), 1)

    def test_recent_gaps_block_predictions(self):
        raw = fixture(1440)
        cutoff = raw.timestamp.iloc[-1] + pd.Timedelta(minutes=1)
        validate_recent_minutes(raw, cutoff, days=1)
        with self.assertRaisesRegex(ValueError, "missing minutes"):
            validate_recent_minutes(raw.drop(index=100), cutoff, days=1)

    def test_live_features_include_latest_closed_candle_without_targets(self):
        raw = fixture()
        cutoff = raw.timestamp.iloc[-1] + pd.Timedelta(minutes=1)
        candles = {name: complete_resample(raw, minutes, cutoff) for name, minutes in TIMEFRAMES.items()}
        features = build_live_features(candles)
        self.assertFalse(any(name.startswith("target_") for name in features.columns))
        self.assertEqual(features.timestamp.iloc[-1], candles["15m"].timestamp.iloc[-1])
        self.assertEqual(features.feature_asof.iloc[-1], features.timestamp.iloc[-1] + pd.Timedelta(minutes=15))


if __name__ == "__main__":
    unittest.main()
