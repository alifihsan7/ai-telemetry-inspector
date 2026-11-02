"""AI Telemetry & Model Unit Economics Inspector - Analytical Data Pipeline.

High-performance data pipeline extracting, transforming, and materializing
operational metrics from 9Router SQLite telemetry into executive JSON marts.
Complies with Sage's canonical PRD specifications.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# WIB Timezone: Asia/Jakarta (UTC+7)
WIB_TZ = timezone(timedelta(hours=7))

# Commercial Retail Frontier Benchmark Pricing (USD per token)
# Standard Anthropic Claude 3.5 Sonnet / OpenAI GPT-4o retail list price:
# $3.00 / MTok input  ($0.000003 / token)
# $15.00 / MTok output ($0.000015 / token)
BENCHMARK_PROMPT_PRICE_PER_TOKEN = 3.00 * 1e-6
BENCHMARK_COMPLETION_PRICE_PER_TOKEN = 15.00 * 1e-6

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def canonical_model_name(raw_model: Optional[str]) -> str:
    """Strip provider slug prefixes (e.g., 'ag/gemini-3.8-flash' -> 'gemini-3.8-flash')."""
    if not raw_model:
        return "unknown"
    model = raw_model.strip()
    if "/" in model:
        return model.split("/", 1)[-1]
    return model


def calculate_benchmark_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Compute hypothetical retail frontier benchmark cost."""
    pt = max(0, prompt_tokens or 0)
    ct = max(0, completion_tokens or 0)
    return (pt * BENCHMARK_PROMPT_PRICE_PER_TOKEN) + (
        ct * BENCHMARK_COMPLETION_PRICE_PER_TOKEN
    )


def calculate_etc(gross_cost: float, total_tokens: int) -> float:
    """Compute Effective Token Cost (ETC) in USD per Million Tokens ($/MTok)."""
    if not total_tokens or total_tokens <= 0:
        return 0.0
    return (float(gross_cost or 0.0) / float(total_tokens)) * 1_000_000.0


def calculate_rom(reasoning_tokens: int, completion_tokens: int) -> float:
    """Compute Reasoning Overhead Multiplier (ROM).

    Formula: reasoning_tokens / (completion_tokens - reasoning_tokens)
    Returns 0.0 if denominator is <= 0.
    """
    rt = max(0, reasoning_tokens or 0)
    ct = max(0, completion_tokens or 0)
    visible_tokens = ct - rt
    if visible_tokens <= 0:
        return 0.0
    return float(rt) / float(visible_tokens)


def calculate_rasi(actual_cost: float, benchmark_cost: float) -> Tuple[float, float]:
    """Compute Routing Arbitrage Savings Index (savings_usd, savings_pct).

    Formula: RASI % = ((benchmark_cost - actual_cost) / benchmark_cost) * 100%
    """
    act = float(actual_cost or 0.0)
    bm = float(benchmark_cost or 0.0)
    savings_usd = bm - act
    if bm <= 0.0:
        return 0.0, 0.0
    savings_pct = (savings_usd / bm) * 100.0
    return savings_usd, savings_pct


def calculate_percentiles(durations: List[float]) -> Dict[str, float]:
    """Calculate empirical percentile latencies P50, P90, P99 and Tail Volatility Index (TVI).

    Formula (PRD Sec 2.3): P_k = d_(ceil(k/100 * N))
    """
    if not durations:
        return {
            "p50_latency_sec": 0.0,
            "p90_latency_sec": 0.0,
            "p99_latency_sec": 0.0,
            "tvi_index": 0.0,
        }

    sorted_d = sorted(durations)
    n = len(sorted_d)

    def get_pk(k: float) -> float:
        idx = max(0, min(math.ceil((k / 100.0) * n) - 1, n - 1))
        return sorted_d[idx]

    p50 = get_pk(50)
    p90 = get_pk(90)
    p99 = get_pk(99)

    tvi = ((p99 - p50) / p50) if p50 > 0.0 else 0.0

    return {
        "p50_latency_sec": round(p50, 4),
        "p90_latency_sec": round(p90, 4),
        "p99_latency_sec": round(p99, 4),
        "tvi_index": round(tvi, 4),
    }


def extract_duration(
    meta_dict: Dict[str, Any],
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    prev_dt: Optional[float],
    curr_dt: float,
    same_conn: bool,
) -> float:
    """Determine operational transaction duration in seconds.

    1. Checks explicit duration/latency in meta if present.
    2. Uses inter-request elapsed time on the same connection if bounded (0.05s - 300s).
    3. Falls back to token throughput physical modeling.
    """
    # 1. Explicit meta checks
    if meta_dict:
        for k in ("duration", "latency", "response_time", "time"):
            v = meta_dict.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        for k in ("duration_ms", "durationMs", "latency_ms", "latencyMs", "responseTimeMs"):
            v = meta_dict.get(k)
            if isinstance(v, (int, float)) and v > 0:
                return float(v) / 1000.0

    # 2. Inter-request timestamp difference on active connection
    if same_conn and prev_dt is not None:
        diff = curr_dt - prev_dt
        if 0.05 <= diff <= 300.0:
            return round(diff, 3)

    # 3. Token-based duration modeling (TTFT + generation + reasoning)
    pt = max(0, prompt_tokens or 0)
    ct = max(0, completion_tokens or 0)
    rt = max(0, reasoning_tokens or 0)
    visible = max(0, ct - rt)

    base_ttft = 0.35 + min(pt * 0.00005, 2.0)
    gen_time = visible / 45.0
    reason_time = rt / 25.0
    return round(base_ttft + gen_time + reason_time, 3)


class TelemetryPipeline:
    """High-performance ETL and metric calculation pipeline for 9Router telemetry."""

    def __init__(
        self,
        db_path: str = "/home/alifihsan/.9router/db/data.sqlite",
        output_dir: str = "/home/alifihsan/ai-telemetry-inspector/data",
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_raw_records(self) -> List[sqlite3.Row]:
        """Fetch all telemetry records from SQLite read-only."""
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    id,
                    timestamp,
                    provider,
                    model,
                    connectionId,
                    apiKey,
                    endpoint,
                    promptTokens,
                    completionTokens,
                    cost,
                    status,
                    tokens,
                    meta
                FROM usageHistory
                ORDER BY id ASC
                """
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def process(self, records: Optional[List[sqlite3.Row]] = None) -> Dict[str, Any]:
        """Execute extraction, analytical aggregation, and JSON artifact materialization."""
        start_time = time.perf_counter()

        if records is None:
            raw_records = self.extract_raw_records()
        else:
            raw_records = records

        total_requests = len(raw_records)

        # Fleet-level accumulators
        fleet_gross_cost = 0.0
        fleet_benchmark_cost = 0.0
        fleet_prompt_tokens = 0
        fleet_completion_tokens = 0
        fleet_cached_tokens = 0
        fleet_reasoning_tokens = 0
        fleet_errors = 0
        fleet_rom_sum = 0.0

        # Model x Provider accumulators
        # key: (canonical_model, provider_slug)
        model_accumulators: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Diurnal 24x7 accumulators (WIB timezone)
        # key: (day_idx 0..6, hour 0..23)
        diurnal_cells: Dict[Tuple[int, int], Dict[str, Any]] = {
            (d, h): {
                "day": DAY_NAMES[d],
                "day_idx": d,
                "hour": h,
                "requests": 0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_cost_usd": 0.0,
            }
            for d in range(7)
            for h in range(24)
        }

        # Error attribution accumulators
        provider_errors: Dict[str, Dict[str, Any]] = {}
        connection_errors: Dict[str, Dict[str, Any]] = {}
        status_distribution: Dict[str, int] = {}

        # Outliers tracking: list of (duration, dict_record)
        outlier_candidates: List[Tuple[float, Dict[str, Any]]] = []

        prev_timestamp: Optional[float] = None
        prev_conn_id: Optional[str] = None

        for row in raw_records:
            rid = row["id"]
            ts_str = row["timestamp"] or ""
            provider = (row["provider"] or "unknown").strip().lower()
            raw_model = row["model"] or "unknown"
            canonical_model = canonical_model_name(raw_model)
            conn_id = row["connectionId"]
            pt = int(row["promptTokens"] or 0)
            ct = int(row["completionTokens"] or 0)
            actual_cost = float(row["cost"] or 0.0)
            status = (row["status"] or "ok").strip()

            # Parse JSON tokens payload
            tokens_str = row["tokens"]
            cached_tokens = 0
            reasoning_tokens = 0
            if tokens_str and tokens_str != "{}":
                try:
                    t_payload = json.loads(tokens_str)
                    cached_tokens = int(
                        t_payload.get("cached_tokens")
                        or t_payload.get("cache_read_input_tokens")
                        or 0
                    )
                    reasoning_tokens = int(t_payload.get("reasoning_tokens") or 0)
                except Exception:
                    pass

            # Parse JSON meta payload
            meta_str = row["meta"]
            meta_dict = {}
            if meta_str and meta_str != "{}":
                try:
                    meta_dict = json.loads(meta_str)
                except Exception:
                    pass

            # Parse timestamps
            dt_utc = None
            dt_wib = None
            curr_ts = 0.0
            if ts_str:
                try:
                    clean_ts = ts_str.replace("Z", "+00:00")
                    dt_utc = datetime.fromisoformat(clean_ts)
                    if dt_utc.tzinfo is None:
                        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                    dt_wib = dt_utc.astimezone(WIB_TZ)
                    curr_ts = dt_utc.timestamp()
                except Exception:
                    pass

            same_conn = (conn_id is not None) and (conn_id == prev_conn_id)
            duration_sec = extract_duration(
                meta_dict=meta_dict,
                prompt_tokens=pt,
                completion_tokens=ct,
                reasoning_tokens=reasoning_tokens,
                prev_dt=prev_timestamp,
                curr_dt=curr_ts,
                same_conn=same_conn,
            )

            prev_timestamp = curr_ts
            prev_conn_id = conn_id

            # Calculations
            bm_cost = calculate_benchmark_cost(pt, ct)
            req_rom = calculate_rom(reasoning_tokens, ct)
            is_error = status.lower() != "ok"

            # Fleet accumulation
            fleet_gross_cost += actual_cost
            fleet_benchmark_cost += bm_cost
            fleet_prompt_tokens += pt
            fleet_completion_tokens += ct
            fleet_cached_tokens += cached_tokens
            fleet_reasoning_tokens += reasoning_tokens
            fleet_rom_sum += req_rom
            if is_error:
                fleet_errors += 1

            # Status distribution
            status_distribution[status] = status_distribution.get(status, 0) + 1

            # Model accumulator
            model_key = (canonical_model, provider)
            if model_key not in model_accumulators:
                model_accumulators[model_key] = {
                    "canonical_model": canonical_model,
                    "provider": provider,
                    "total_requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "cached_tokens": 0,
                    "gross_cost": 0.0,
                    "benchmark_cost": 0.0,
                    "rom_sum": 0.0,
                    "ok_count": 0,
                    "durations": [],
                }
            m_acc = model_accumulators[model_key]
            m_acc["total_requests"] += 1
            m_acc["prompt_tokens"] += pt
            m_acc["completion_tokens"] += ct
            m_acc["reasoning_tokens"] += reasoning_tokens
            m_acc["cached_tokens"] += cached_tokens
            m_acc["gross_cost"] += actual_cost
            m_acc["benchmark_cost"] += bm_cost
            m_acc["rom_sum"] += req_rom
            if not is_error:
                m_acc["ok_count"] += 1
            m_acc["durations"].append(duration_sec)

            # Diurnal accumulation
            if dt_wib is not None:
                d_key = (dt_wib.weekday(), dt_wib.hour)
                if d_key in diurnal_cells:
                    cell = diurnal_cells[d_key]
                    cell["requests"] += 1
                    cell["total_tokens"] += pt + ct
                    cell["prompt_tokens"] += pt
                    cell["completion_tokens"] += ct
                    cell["total_cost_usd"] += actual_cost

            # Error attribution accumulation
            if provider not in provider_errors:
                provider_errors[provider] = {
                    "provider": provider,
                    "total_requests": 0,
                    "error_count": 0,
                    "error_rate_pct": 0.0,
                    "status_distribution": {},
                }
            p_entry = provider_errors[provider]
            p_entry["total_requests"] += 1
            p_entry["status_distribution"][status] = (
                p_entry["status_distribution"].get(status, 0) + 1
            )
            if is_error:
                p_entry["error_count"] += 1

            cid_str = conn_id or "unassigned"
            if cid_str not in connection_errors:
                connection_errors[cid_str] = {
                    "connection_id": cid_str,
                    "provider": provider,
                    "total_requests": 0,
                    "error_count": 0,
                    "error_rate_pct": 0.0,
                    "status_distribution": {},
                }
            c_entry = connection_errors[cid_str]
            c_entry["total_requests"] += 1
            c_entry["status_distribution"][status] = (
                c_entry["status_distribution"].get(status, 0) + 1
            )
            if is_error:
                c_entry["error_count"] += 1

            # Outlier candidate
            outlier_candidates.append(
                (
                    duration_sec,
                    {
                        "request_id": rid,
                        "timestamp_utc": dt_utc.isoformat() if dt_utc else ts_str,
                        "timestamp_wib": dt_wib.isoformat() if dt_wib else "",
                        "duration_sec": round(duration_sec, 3),
                        "model": canonical_model,
                        "raw_model": raw_model,
                        "provider": provider,
                        "connection_id": conn_id,
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "reasoning_tokens": reasoning_tokens,
                        "total_tokens": pt + ct,
                        "cost_usd": round(actual_cost, 6),
                        "status": status,
                    },
                )
            )

        # -------------------------------------------------------------------------
        # 1. Summary Artifact
        # -------------------------------------------------------------------------
        fleet_total_tokens = fleet_prompt_tokens + fleet_completion_tokens
        rasi_savings_usd, rasi_savings_pct = calculate_rasi(
            fleet_gross_cost, fleet_benchmark_cost
        )
        blended_etc = calculate_etc(fleet_gross_cost, fleet_total_tokens)
        error_rate_pct = (
            (fleet_errors / total_requests) * 100.0 if total_requests > 0 else 0.0
        )
        ok_rate_pct = (
            ((total_requests - fleet_errors) / total_requests) * 100.0
            if total_requests > 0
            else 100.0
        )
        avg_rom = (
            (fleet_rom_sum / total_requests) if total_requests > 0 else 0.0
        )
        visible_completion = max(0, fleet_completion_tokens - fleet_reasoning_tokens)
        fleet_rom_ratio = (
            (fleet_reasoning_tokens / visible_completion)
            if visible_completion > 0
            else 0.0
        )
        crdr = (
            1.0 - (fleet_gross_cost / fleet_benchmark_cost)
            if fleet_benchmark_cost > 0.0
            else 0.0
        )

        summary_data = {
            "total_requests": total_requests,
            "gross_spend_usd": round(fleet_gross_cost, 4),
            "benchmark_frontier_spend_usd": round(fleet_benchmark_cost, 4),
            "rasi_savings_usd": round(rasi_savings_usd, 4),
            "rasi_savings_pct": round(rasi_savings_pct, 2),
            "blended_etc_per_mtok": round(blended_etc, 4),
            "total_tokens": fleet_total_tokens,
            "prompt_tokens": fleet_prompt_tokens,
            "completion_tokens": fleet_completion_tokens,
            "cached_tokens": fleet_cached_tokens,
            "reasoning_tokens": fleet_reasoning_tokens,
            "visible_completion_tokens": visible_completion,
            "fleet_rom_ratio": round(fleet_rom_ratio, 4),
            "avg_rom_per_request": round(avg_rom, 4),
            "cache_realized_discount_ratio": round(crdr, 4),
            "total_errors": fleet_errors,
            "error_rate_pct": round(error_rate_pct, 4),
            "status_ok_rate_pct": round(ok_rate_pct, 4),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # -------------------------------------------------------------------------
        # 2. Model Frontier Artifact
        # -------------------------------------------------------------------------
        model_frontier_data: List[Dict[str, Any]] = []
        for (c_model, prov), m_acc in model_accumulators.items():
            m_reqs = m_acc["total_requests"]
            m_pt = m_acc["prompt_tokens"]
            m_ct = m_acc["completion_tokens"]
            m_rt = m_acc["reasoning_tokens"]
            m_cacht = m_acc["cached_tokens"]
            m_tot = m_pt + m_ct
            m_gross = m_acc["gross_cost"]
            m_bm = m_acc["benchmark_cost"]
            m_ok = m_acc["ok_count"]

            m_rasi_usd, m_rasi_pct = calculate_rasi(m_gross, m_bm)
            m_etc = calculate_etc(m_gross, m_tot)
            m_visible = max(0, m_ct - m_rt)
            m_rom_ratio = (float(m_rt) / m_visible) if m_visible > 0 else 0.0
            m_avg_rom = (m_acc["rom_sum"] / m_reqs) if m_reqs > 0 else 0.0
            m_ok_pct = ((m_ok / m_reqs) * 100.0) if m_reqs > 0 else 100.0

            pctiles = calculate_percentiles(m_acc["durations"])

            model_frontier_data.append(
                {
                    "model": c_model,
                    "provider": prov,
                    "total_requests": m_reqs,
                    "prompt_tokens": m_pt,
                    "completion_tokens": m_ct,
                    "reasoning_tokens": m_rt,
                    "cached_tokens": m_cacht,
                    "total_tokens": m_tot,
                    "gross_cost_usd": round(m_gross, 4),
                    "benchmark_cost_usd": round(m_bm, 4),
                    "rasi_savings_usd": round(m_rasi_usd, 4),
                    "rasi_savings_pct": round(m_rasi_pct, 2),
                    "etc_per_mtok": round(m_etc, 4),
                    "rom_ratio": round(m_rom_ratio, 4),
                    "avg_rom": round(m_avg_rom, 4),
                    "p50_latency_sec": pctiles["p50_latency_sec"],
                    "p90_latency_sec": pctiles["p90_latency_sec"],
                    "p99_latency_sec": pctiles["p99_latency_sec"],
                    "tvi_index": pctiles["tvi_index"],
                    "status_ok_rate_pct": round(m_ok_pct, 2),
                }
            )

        # Sort model frontier by total_requests descending
        model_frontier_data.sort(key=lambda x: x["total_requests"], reverse=True)

        # -------------------------------------------------------------------------
        # 3. Diurnal Matrix Artifact (24h x 7d in WIB)
        # -------------------------------------------------------------------------
        # Prepare 2D grid matrices for fast visualization
        grid_requests = [[0] * 24 for _ in range(7)]
        grid_tokens = [[0] * 24 for _ in range(7)]
        grid_cost = [[0.0] * 24 for _ in range(7)]

        flat_cells = []
        for d in range(7):
            for h in range(24):
                c = diurnal_cells[(d, h)]
                c["total_cost_usd"] = round(c["total_cost_usd"], 4)
                flat_cells.append(c)
                grid_requests[d][h] = c["requests"]
                grid_tokens[d][h] = c["total_tokens"]
                grid_cost[d][h] = c["total_cost_usd"]

        diurnal_matrix_data = {
            "timezone": "Asia/Jakarta (WIB, UTC+7)",
            "days": DAY_NAMES,
            "hours": list(range(24)),
            "cells": flat_cells,
            "grid": {
                "requests": grid_requests,
                "tokens": grid_tokens,
                "cost_usd": grid_cost,
            },
        }

        # -------------------------------------------------------------------------
        # 4. Error Attribution Artifact
        # -------------------------------------------------------------------------
        for p_val in provider_errors.values():
            reqs = p_val["total_requests"]
            errs = p_val["error_count"]
            p_val["error_rate_pct"] = (
                round((errs / reqs) * 100.0, 2) if reqs > 0 else 0.0
            )

        for c_val in connection_errors.values():
            reqs = c_val["total_requests"]
            errs = c_val["error_count"]
            c_val["error_rate_pct"] = (
                round((errs / reqs) * 100.0, 2) if reqs > 0 else 0.0
            )

        error_attribution_data = {
            "total_requests": total_requests,
            "total_errors": fleet_errors,
            "overall_error_rate_pct": round(error_rate_pct, 4),
            "status_distribution": status_distribution,
            "by_provider": provider_errors,
            "by_connection_id": connection_errors,
        }

        # -------------------------------------------------------------------------
        # 5. Top Outliers Artifact (Top 25 P99+ Tail Latencies)
        # -------------------------------------------------------------------------
        outlier_candidates.sort(key=lambda x: x[0], reverse=True)
        top_25_raw = outlier_candidates[:25]
        top_outliers_data = []
        for rank, (_, item) in enumerate(top_25_raw, start=1):
            entry = {"rank": rank, **item}
            top_outliers_data.append(entry)

        # -------------------------------------------------------------------------
        # Write Files to disk atomically
        # -------------------------------------------------------------------------
        artifacts = {
            "summary.json": summary_data,
            "model_frontier.json": model_frontier_data,
            "diurnal_matrix.json": diurnal_matrix_data,
            "error_attribution.json": error_attribution_data,
            "top_outliers.json": top_outliers_data,
        }

        for filename, data in artifacts.items():
            filepath = self.output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        elapsed = time.perf_counter() - start_time
        return {
            "total_requests": total_requests,
            "elapsed_seconds": round(elapsed, 4),
            "output_dir": str(self.output_dir),
            "artifacts": list(artifacts.keys()),
            "summary": summary_data,
        }


def run() -> None:
    """CLI runner executing the telemetry pipeline."""
    db_path = os.environ.get(
        "ROUTER_DB_PATH", "/home/alifihsan/.9router/db/data.sqlite"
    )
    output_dir = os.environ.get(
        "INSPECTOR_DATA_DIR", "/home/alifihsan/ai-telemetry-inspector/data"
    )

    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print("================================================================================")
    print("AI TELEMETRY & MODEL UNIT ECONOMICS INSPECTOR - DATA PIPELINE")
    print("================================================================================")
    print(f"Source Database: {db_path}")
    print(f"Output Directory: {output_dir}")
    print("Extracting & Materializing Analytics...")

    pipeline = TelemetryPipeline(db_path=db_path, output_dir=output_dir)
    res = pipeline.process()

    s = res["summary"]
    print("\n------------------------- EXECUTION SUMMARY -------------------------")
    print(f"Total Requests Processed: {res['total_requests']:,}")
    print(f"Elapsed Time:             {res['elapsed_seconds']:.3f} seconds")
    print(f"Total Gross Spend:        ${s['gross_spend_usd']:,.4f}")
    print(f"Benchmark Retail Spend:   ${s['benchmark_frontier_spend_usd']:,.4f}")
    print(f"RASI Arbitrage Savings:   ${s['rasi_savings_usd']:,.4f} ({s['rasi_savings_pct']:.2f}%)")
    print(f"Blended ETC:              ${s['blended_etc_per_mtok']:.4f}/MTok")
    print(f"Total Prompt Tokens:      {s['prompt_tokens']:,}")
    print(f"Total Completion Tokens:  {s['completion_tokens']:,}")
    print(f"Total Reasoning Tokens:   {s['reasoning_tokens']:,}")
    print(f"Total Cached Tokens:      {s['cached_tokens']:,}")
    print(f"Status OK Rate:           {s['status_ok_rate_pct']:.2f}%")
    print("---------------------------------------------------------------------")
    print("Generated Artifacts:")
    for a in res["artifacts"]:
        print(f"  • {output_dir}/{a}")
    print("================================================================================")


if __name__ == "__main__":
    run()
