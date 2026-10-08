"""Regression gate: fail CI when quality drops below the blessed baseline.

Usage in CI:
    python -m src.eval.regression --report eval_report.json [--update-baseline]

The gate compares the current eval report against `configs/baseline.json`
(min pass rate, min avg groundedness, max avg latency). Exit code 1 on
regression so CI fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List


def check_regression(report: Dict, baseline: Dict) -> Dict[str, object]:
    failures: List[str] = []
    pr = report.get("pass_rate", 0)
    if pr < baseline.get("min_pass_rate", 0.8):
        failures.append(f"pass_rate {pr} < {baseline['min_pass_rate']}")
    ag = report.get("avg_groundedness", 0)
    if ag < baseline.get("min_groundedness", 0.4):
        failures.append(f"avg_groundedness {ag} < {baseline['min_groundedness']}")
    lat = report.get("avg_latency_ms", 0)
    if lat > baseline.get("max_latency_ms", 5000):
        failures.append(f"avg_latency_ms {lat} > {baseline['max_latency_ms']}")
    return {"passed": not failures, "failures": failures}


def main(argv=None) -> int:  # pragma: no cover - thin CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--baseline", default="configs/baseline.json")
    args = ap.parse_args(argv)
    with open(args.report) as f:
        report = json.load(f)
    with open(args.baseline) as f:
        baseline = json.load(f)
    res = check_regression(report, baseline)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
