"""Eval package: golden datasets, LLM-as-judge, regression gates."""

from .dataset import GoldenExample, load_golden, save_golden
from .judge import EvalCaseResult, JudgeResult, judge_groundedness, run_eval
from .regression import check_regression

__all__ = [
    "GoldenExample", "load_golden", "save_golden",
    "EvalCaseResult", "JudgeResult", "judge_groundedness", "run_eval",
    "check_regression",
]
