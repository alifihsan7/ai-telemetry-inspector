"""Unit Test Suite for AI Telemetry & Model Unit Economics Inspector Pipeline.

Tests mathematical formula integrity, zero-division guards, edge case handling,
in-memory SQLite data processing, and output JSON artifact schema validation.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pipeline import (
    BENCHMARK_COMPLETION_PRICE_PER_TOKEN,
    BENCHMARK_PROMPT_PRICE_PER_TOKEN,
    DAY_NAMES,
    TelemetryPipeline,
    calculate_benchmark_cost,
    calculate_etc,
    calculate_percentiles,
    calculate_rasi,
    calculate_rom,
    canonical_model_name,
    extract_duration,
)


class TestMathematicalFormulas(unittest.TestCase):
    """Test mathematical formulas and zero division guards."""

    def test_canonical_model_name(self):
        self.assertEqual(canonical_model_name("ag/gemini-3.8-flash"), "gemini-3.8-flash")
        self.assertEqual(canonical_model_name("cx/gpt-6.1-sol"), "gpt-6.1-sol")
        self.assertEqual(canonical_model_name("deepseek-v4.1-flash-free"), "deepseek-v4.1-flash-free")
        self.assertEqual(canonical_model_name(""), "unknown")
        self.assertEqual(canonical_model_name(None), "unknown")

    def test_calculate_benchmark_cost(self):
        # 1M prompt tokens = $3.00, 1M completion tokens = $15.00
        cost = calculate_benchmark_cost(1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 18.00, places=6)

        # Zero tokens
        self.assertEqual(calculate_benchmark_cost(0, 0), 0.0)
        # None / negative inputs
        self.assertEqual(calculate_benchmark_cost(None, -5), 0.0)

    def test_calculate_etc(self):
        # $10 for 10M tokens -> $1.00 / MTok
        etc = calculate_etc(10.0, 10_000_000)
        self.assertAlmostEqual(etc, 1.0, places=4)

        # Zero division guard
        self.assertEqual(calculate_etc(10.0, 0), 0.0)
        self.assertEqual(calculate_etc(0.0, 0), 0.0)
        self.assertEqual(calculate_etc(5.0, -100), 0.0)
        self.assertEqual(calculate_etc(None, None), 0.0)

    def test_calculate_rom(self):
        # 300 reasoning, 400 completion -> 100 visible -> ROM = 300 / 100 = 3.0
        rom = calculate_rom(300, 400)
        self.assertAlmostEqual(rom, 3.0, places=4)

        # Reasoning equals completion -> visible = 0 -> Zero division guard returns 0.0
        self.assertEqual(calculate_rom(400, 400), 0.0)

        # Reasoning > completion -> visible < 0 -> Returns 0.0
        self.assertEqual(calculate_rom(500, 400), 0.0)

        # Zero reasoning -> ROM = 0.0
        self.assertEqual(calculate_rom(0, 400), 0.0)
        self.assertEqual(calculate_rom(None, 0), 0.0)

    def test_calculate_rasi(self):
        # Benchmark = $10.0, Actual = $2.0 -> Savings = $8.0 (80.0%)
        savings_usd, savings_pct = calculate_rasi(2.0, 10.0)
        self.assertAlmostEqual(savings_usd, 8.0, places=4)
        self.assertAlmostEqual(savings_pct, 80.0, places=2)

        # Zero benchmark spend -> Zero division guard returns (0.0, 0.0)
        savings_usd, savings_pct = calculate_rasi(5.0, 0.0)
        self.assertEqual(savings_usd, 0.0)
        self.assertEqual(savings_pct, 0.0)

        # Negative / None handling
        self.assertEqual(calculate_rasi(None, 0.0), (0.0, 0.0))

    def test_calculate_percentiles(self):
        # 100 observations: 1.0 to 100.0
        durations = [float(i) for i in range(1, 101)]
        res = calculate_percentiles(durations)

        # P50 = 50.0, P90 = 90.0, P99 = 99.0
        self.assertAlmostEqual(res["p50_latency_sec"], 50.0, places=2)
        self.assertAlmostEqual(res["p90_latency_sec"], 90.0, places=2)
        self.assertAlmostEqual(res["p99_latency_sec"], 99.0, places=2)

        # TVI = (99.0 - 50.0) / 50.0 = 49.0 / 50.0 = 0.98
        self.assertAlmostEqual(res["tvi_index"], 0.98, places=4)

        # Empty list handling
        empty_res = calculate_percentiles([])
        self.assertEqual(empty_res["p50_latency_sec"], 0.0)
        self.assertEqual(empty_res["p90_latency_sec"], 0.0)
        self.assertEqual(empty_res["p99_latency_sec"], 0.0)
        self.assertEqual(empty_res["tvi_index"], 0.0)

        # Single element
        single_res = calculate_percentiles([5.0])
        self.assertEqual(single_res["p50_latency_sec"], 5.0)
        self.assertEqual(single_res["p90_latency_sec"], 5.0)
        self.assertEqual(single_res["p99_latency_sec"], 5.0)
        self.assertEqual(single_res["tvi_index"], 0.0)

        # All zeros -> TVI zero division guard
        zero_res = calculate_percentiles([0.0, 0.0, 0.0])
        self.assertEqual(zero_res["p50_latency_sec"], 0.0)
        self.assertEqual(zero_res["tvi_index"], 0.0)

    def test_extract_duration(self):
        # Explicit meta with duration
        dur = extract_duration(
            meta_dict={"duration": 4.5},
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=0,
            prev_dt=None,
            curr_dt=10.0,
            same_conn=False,
        )
        self.assertEqual(dur, 4.5)

        # Explicit meta with duration_ms
        dur_ms = extract_duration(
            meta_dict={"duration_ms": 2500},
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=0,
            prev_dt=None,
            curr_dt=10.0,
            same_conn=False,
        )
        self.assertEqual(dur_ms, 2.5)

        # Consecutive timestamp diff on same connection
        dur_diff = extract_duration(
            meta_dict={},
            prompt_tokens=100,
            completion_tokens=50,
            reasoning_tokens=0,
            prev_dt=100.0,
            curr_dt=105.25,
            same_conn=True,
        )
        self.assertAlmostEqual(dur_diff, 5.25, places=2)

        # Diff too large (>300s) -> fallback to token throughput
        dur_fallback = extract_duration(
            meta_dict={},
            prompt_tokens=2000,
            completion_tokens=450,
            reasoning_tokens=50,
            prev_dt=100.0,
            curr_dt=5000.0,
            same_conn=True,
        )
        self.assertGreater(dur_fallback, 0.0)
        self.assertLess(dur_fallback, 60.0)


class TestPipelineWithMockDatabase(unittest.TestCase):
    """Test full pipeline execution on a synthetic SQLite database."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test_data.sqlite")
        self.output_dir = str(Path(self.temp_dir.name) / "data")

        # Create mock 9router database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE usageHistory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                connectionId TEXT,
                apiKey TEXT,
                endpoint TEXT,
                promptTokens INTEGER DEFAULT 0,
                completionTokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                status TEXT,
                tokens TEXT,
                meta TEXT
            )
            """
        )

        # Insert representative test records
        # Record 1: Normal success request (Antigravity gemini-3.8-flash)
        c.execute(
            """
            INSERT INTO usageHistory (timestamp, provider, model, connectionId, promptTokens, completionTokens, cost, status, tokens, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-11-02T10:00:00.000Z",  # Monday 17:00 WIB
                "antigravity",
                "ag/gemini-3.8-flash",
                "conn-01",
                1000,
                200,
                0.001,
                "ok",
                json.dumps({"prompt_tokens": 1000, "completion_tokens": 200, "cached_tokens": 500, "reasoning_tokens": 50}),
                json.dumps({"duration": 2.5}),
            ),
        )

        # Record 2: Reasoning heavy request (Opencode deepseek-r1)
        c.execute(
            """
            INSERT INTO usageHistory (timestamp, provider, model, connectionId, promptTokens, completionTokens, cost, status, tokens, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-11-02T11:30:00.000Z",  # Monday 18:30 WIB
                "opencode",
                "oc/deepseek-r1",
                "conn-02",
                5000,
                2000,
                0.015,
                "ok",
                json.dumps({"prompt_tokens": 5000, "completion_tokens": 2000, "cached_tokens": 0, "reasoning_tokens": 1500}),
                json.dumps({"duration": 45.0}),
            ),
        )

        # Record 3: Error transaction (429 RateLimit on Kiro)
        c.execute(
            """
            INSERT INTO usageHistory (timestamp, provider, model, connectionId, promptTokens, completionTokens, cost, status, tokens, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-11-03T01:15:00.000Z",  # Tuesday 08:15 WIB
                "kiro",
                "claude-sonnet-4.5",
                "conn-03",
                500,
                0,
                0.0,
                "429 RateLimit",
                json.dumps({"prompt_tokens": 500, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}),
                json.dumps({"duration": 0.5}),
            ),
        )

        # Record 4: Error transaction (503 ProviderDown on Kiro)
        c.execute(
            """
            INSERT INTO usageHistory (timestamp, provider, model, connectionId, promptTokens, completionTokens, cost, status, tokens, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-11-03T01:20:00.000Z",  # Tuesday 08:20 WIB
                "kiro",
                "claude-sonnet-4.5",
                "conn-03",
                600,
                0,
                0.0,
                "503 Service Unavailable",
                json.dumps({"prompt_tokens": 600, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}),
                json.dumps({"duration": 1.2}),
            ),
        )

        conn.commit()
        conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_execution_and_artifacts(self):
        pipeline = TelemetryPipeline(db_path=self.db_path, output_dir=self.output_dir)
        res = pipeline.process()

        self.assertEqual(res["total_requests"], 4)
        self.assertIn("summary.json", res["artifacts"])
        self.assertIn("model_frontier.json", res["artifacts"])
        self.assertIn("diurnal_matrix.json", res["artifacts"])
        self.assertIn("error_attribution.json", res["artifacts"])
        self.assertIn("top_outliers.json", res["artifacts"])

        # 1. Validate summary.json
        with open(Path(self.output_dir) / "summary.json") as f:
            summary = json.load(f)
        self.assertEqual(summary["total_requests"], 4)
        self.assertEqual(summary["total_errors"], 2)
        self.assertAlmostEqual(summary["error_rate_pct"], 50.0, places=2)
        self.assertAlmostEqual(summary["status_ok_rate_pct"], 50.0, places=2)
        self.assertEqual(summary["prompt_tokens"], 7100)
        self.assertEqual(summary["completion_tokens"], 2200)
        self.assertEqual(summary["reasoning_tokens"], 1550)
        self.assertEqual(summary["cached_tokens"], 500)
        self.assertGreater(summary["gross_spend_usd"], 0.0)
        self.assertGreater(summary["benchmark_frontier_spend_usd"], 0.0)
        self.assertIn("rasi_savings_pct", summary)
        self.assertIn("blended_etc_per_mtok", summary)

        # 2. Validate model_frontier.json
        with open(Path(self.output_dir) / "model_frontier.json") as f:
            mf = json.load(f)
        self.assertIsInstance(mf, list)
        self.assertGreater(len(mf), 0)

        models = {item["model"]: item for item in mf}
        self.assertIn("gemini-3.8-flash", models)
        self.assertIn("deepseek-r1", models)
        self.assertIn("claude-sonnet-4.5", models)

        # Canonical names stripped of 'ag/' and 'oc/'
        self.assertNotIn("ag/gemini-3.8-flash", models)
        self.assertNotIn("oc/deepseek-r1", models)

        # Check deepseek-r1 ROM
        ds = models["deepseek-r1"]
        # completion 2000, reasoning 1500 -> visible 500 -> ROM = 1500 / 500 = 3.0
        self.assertAlmostEqual(ds["rom_ratio"], 3.0, places=2)
        self.assertEqual(ds["p50_latency_sec"], 45.0)

        # 3. Validate diurnal_matrix.json
        with open(Path(self.output_dir) / "diurnal_matrix.json") as f:
            dm = json.load(f)
        self.assertEqual(dm["timezone"], "Asia/Jakarta (WIB, UTC+7)")
        self.assertEqual(len(dm["days"]), 7)
        self.assertEqual(len(dm["hours"]), 24)
        self.assertEqual(len(dm["cells"]), 168)
        self.assertEqual(len(dm["grid"]["requests"]), 7)
        self.assertEqual(len(dm["grid"]["requests"][0]), 24)

        # Record 1 was at 10:00 UTC Monday -> 17:00 WIB Monday (day_idx=0, hour=17)
        monday_17 = [c for c in dm["cells"] if c["day_idx"] == 0 and c["hour"] == 17][0]
        self.assertGreaterEqual(monday_17["requests"], 1)

        # 4. Validate error_attribution.json
        with open(Path(self.output_dir) / "error_attribution.json") as f:
            ea = json.load(f)
        self.assertEqual(ea["total_errors"], 2)
        self.assertIn("kiro", ea["by_provider"])
        self.assertEqual(ea["by_provider"]["kiro"]["error_count"], 2)
        self.assertEqual(ea["by_provider"]["kiro"]["error_rate_pct"], 100.0)
        self.assertIn("conn-03", ea["by_connection_id"])
        self.assertEqual(ea["by_connection_id"]["conn-03"]["error_count"], 2)

        # 5. Validate top_outliers.json
        with open(Path(self.output_dir) / "top_outliers.json") as f:
            to = json.load(f)
        self.assertIsInstance(to, list)
        self.assertEqual(len(to), 4)
        # Rank 1 must be highest duration (deepseek-r1 with duration 45.0)
        self.assertEqual(to[0]["rank"], 1)
        self.assertEqual(to[0]["model"], "deepseek-r1")
        self.assertEqual(to[0]["duration_sec"], 45.0)
        self.assertIn("prompt_tokens", to[0])
        self.assertIn("completion_tokens", to[0])
        self.assertIn("connection_id", to[0])


class TestEmptyDatabaseHandling(unittest.TestCase):
    """Test pipeline resiliency against an empty database."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "empty_data.sqlite")
        self.output_dir = str(Path(self.temp_dir.name) / "data")

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE usageHistory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                provider TEXT,
                model TEXT,
                connectionId TEXT,
                apiKey TEXT,
                endpoint TEXT,
                promptTokens INTEGER DEFAULT 0,
                completionTokens INTEGER DEFAULT 0,
                cost REAL DEFAULT 0,
                status TEXT,
                tokens TEXT,
                meta TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_database_no_crash(self):
        pipeline = TelemetryPipeline(db_path=self.db_path, output_dir=self.output_dir)
        res = pipeline.process()

        self.assertEqual(res["total_requests"], 0)
        s = res["summary"]
        self.assertEqual(s["total_requests"], 0)
        self.assertEqual(s["gross_spend_usd"], 0.0)
        self.assertEqual(s["benchmark_frontier_spend_usd"], 0.0)
        self.assertEqual(s["rasi_savings_pct"], 0.0)
        self.assertEqual(s["blended_etc_per_mtok"], 0.0)
        self.assertEqual(s["error_rate_pct"], 0.0)
        self.assertEqual(s["status_ok_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
