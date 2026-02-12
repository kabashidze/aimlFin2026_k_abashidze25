"""
Task 3 — Web server log analysis (DDoS intervals) using regression.

Usage (from repo root):
    python task_3/detect_ddos_regression.py --log task_3/k_abashidze25_12847_server.log --tz "+04:00"

Outputs:
- Printed DDoS time intervals (start/end timestamps)
- Saved plots into task_3/:
    - requests_per_min.png
    - fitted_vs_actual.png
    - residuals.png
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# Example log line (Apache-like custom):
# 69.166.180.140 - - [2024-03-22 18:00:16+04:00] "POST /usr/register HTTP/1.0" 200 4964 "-" "UA..." 4004
LOG_RE = re.compile(
    r'^(?P<ip>\S+)\s+-\s+-\s+\[(?P<ts>[0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2}\+\d{2}:\d{2})\]\s+'
    r'"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+HTTP/(?P<httpver>[0-9.]+)"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\d+)\s+"(?P<ref>[^"]*)"\s+"(?P<ua>[^"]*)"\s+(?P<rt>\d+)\s*$'
)

@dataclass(frozen=True)
class LogEvent:
    ip: str
    ts: datetime
    method: str
    path: str
    status: int
    size: int
    rt: int  # response time (as given in log; unit depends on generator)

def parse_log(path: Path) -> List[LogEvent]:
    events: List[LogEvent] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LOG_RE.match(line)
            if not m:
                bad += 1
                continue
            ts = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S%z")
            events.append(
                LogEvent(
                    ip=m.group("ip"),
                    ts=ts,
                    method=m.group("method"),
                    path=m.group("path"),
                    status=int(m.group("status")),
                    size=int(m.group("size")),
                    rt=int(m.group("rt")),
                )
            )
    if len(events) == 0:
        raise ValueError("No parsable events. Check the regex or log format.")
    if bad > 0:
        print(f"[WARN] Skipped {bad} unparsable lines (format mismatch).")
    return events

def floor_to_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)

def aggregate_per_minute(events: List[LogEvent]) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
    # Build minute buckets
    minutes = [floor_to_minute(e.ts) for e in events]
    min_t = min(minutes)
    max_t = max(minutes)
    # inclusive range of minutes
    n = int((max_t - min_t).total_seconds() // 60) + 1
    timeline = [min_t + timedelta(minutes=i) for i in range(n)]

    idx = {t: i for i, t in enumerate(timeline)}
    counts = np.zeros(n, dtype=int)
    uniq_ips = [set() for _ in range(n)]

    for e in events:
        i = idx[floor_to_minute(e.ts)]
        counts[i] += 1
        uniq_ips[i].add(e.ip)

    uniq_ip_counts = np.array([len(s) for s in uniq_ips], dtype=int)
    t = np.arange(n, dtype=float)  # time index in minutes
    return t, counts.astype(float), timeline, uniq_ip_counts.astype(float)

def fit_baseline_regression(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Regression model for baseline traffic.
    We use polynomial regression (degree=2) to capture mild trends without overfitting.
    """
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", LinearRegression()),
    ])
    # Use log1p to stabilize variance typical for count data
    y_log = np.log1p(y)
    model.fit(t.reshape(-1, 1), y_log)
    yhat_log = model.predict(t.reshape(-1, 1))
    yhat = np.expm1(yhat_log)
    return yhat

def detect_attack_intervals(timeline: List[datetime], y: np.ndarray, yhat: np.ndarray,
                            z_thresh: float = 3.0, min_len_minutes: int = 2) -> List[Tuple[datetime, datetime]]:
    """
    Detect attack minutes based on positive residual z-scores.
    Then merge consecutive flagged minutes into intervals.
    """
    resid = y - yhat
    # robust-ish standardization
    mu = np.median(resid)
    sigma = np.median(np.abs(resid - mu)) * 1.4826  # MAD -> std approx
    if sigma <= 1e-9:
        sigma = np.std(resid) + 1e-9
    z = (resid - mu) / sigma

    flagged = z >= z_thresh

    intervals: List[Tuple[datetime, datetime]] = []
    start = None
    for i, is_on in enumerate(flagged):
        if is_on and start is None:
            start = i
        if (not is_on) and start is not None:
            end = i - 1
            if (end - start + 1) >= min_len_minutes:
                intervals.append((timeline[start], timeline[end] + timedelta(minutes=1)))
            start = None
    if start is not None:
        end = len(flagged) - 1
        if (end - start + 1) >= min_len_minutes:
            intervals.append((timeline[start], timeline[end] + timedelta(minutes=1)))
    return intervals

def save_plots(outdir: Path, timeline: List[datetime], y: np.ndarray, yhat: np.ndarray, uniq_ip: np.ndarray) -> None:
    times = np.array(timeline)

    # Plot 1: Requests per minute
    plt.figure()
    plt.plot(times, y, label="Requests/min")
    plt.plot(times, yhat, label="Regression baseline (fit)", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.title("Requests per minute with regression baseline")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "requests_per_min.png", dpi=200)
    plt.close()

    # Plot 2: Fitted vs actual
    plt.figure()
    plt.scatter(yhat, y)
    plt.xlabel("Predicted (baseline) requests/min")
    plt.ylabel("Actual requests/min")
    plt.title("Actual vs Predicted (baseline)")
    plt.tight_layout()
    plt.savefig(outdir / "fitted_vs_actual.png", dpi=200)
    plt.close()

    # Plot 3: Residuals + unique IPs (separate y-axis is avoided; we show as two lines)
    resid = y - yhat
    plt.figure()
    plt.plot(times, resid, label="Residual (actual - baseline)")
    plt.plot(times, uniq_ip, label="Unique IPs/min", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Residuals and unique IPs per minute")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "residuals.png", dpi=200)
    plt.close()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to the server.log file")
    ap.add_argument("--z", type=float, default=3.0, help="Z-score threshold for attack detection (default 3.0)")
    ap.add_argument("--min_len", type=int, default=2, help="Minimum consecutive minutes to form an interval")
    args = ap.parse_args()

    log_path = Path(args.log)
    outdir = log_path.parent

    events = parse_log(log_path)
    t, y, timeline, uniq_ip = aggregate_per_minute(events)

    yhat = fit_baseline_regression(t, y)
    intervals = detect_attack_intervals(timeline, y, yhat, z_thresh=args.z, min_len_minutes=args.min_len)

    print("\nDetected DDoS interval(s):")
    if not intervals:
        print("  (none detected with current parameters)")
    for s, e in intervals:
        print(f"  - {s.isoformat()}  ->  {e.isoformat()}")

    save_plots(outdir, timeline, y, yhat, uniq_ip)
    print("\nSaved plots: requests_per_min.png, fitted_vs_actual.png, residuals.png")

if __name__ == "__main__":
    main()
