"""Detect-eval framework: metrics extraction, comparison, and report archiving."""

from .compare import compute_delta, load_eval_config, resolve_baseline, summarize_metrics
from .metrics import build_metrics_doc, empty_class_metrics, extract_from_ultralytics
from .report import write_eval_run

__all__ = [
    "load_eval_config",
    "resolve_baseline",
    "summarize_metrics",
    "compute_delta",
    "build_metrics_doc",
    "empty_class_metrics",
    "extract_from_ultralytics",
    "write_eval_run",
]
