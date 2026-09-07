from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

_PKG_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EVAL_CONFIG = _PKG_ROOT / "config" / "eval.yaml"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def _resolve_class_groups(meta: dict) -> tuple[list[str], list[str]]:
    focus = meta.get("focus_classes")
    base = meta.get("base_classes")
    if focus is not None and base is not None:
        return list(focus), list(base)

    wave = meta.get("wave")
    config_path = Path(meta.get("eval_config", _DEFAULT_EVAL_CONFIG))
    if wave and config_path.is_file():
        cfg = yaml.safe_load(config_path.read_text())
        focus = list(cfg.get("focus_classes_by_wave", {}).get(wave, []))
        base = list(cfg.get("base_classes", []))
        return focus, base

    return list(focus or []), list(base or [])


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _class_row(name: str, metrics: dict | None) -> list[str]:
    if metrics is None:
        return [name, "N/A", "N/A", "N/A", "N/A"]
    return [
        name,
        _fmt(metrics.get("ap50")),
        _fmt(metrics.get("ap50_95")),
        _fmt(metrics.get("precision")),
        _fmt(metrics.get("recall")),
    ]


def _focus_empty_message(meta: dict) -> str:
    if meta.get("wave_known") is False:
        return f"(none — unknown wave {meta.get('wave', '?')})"
    return "(none)"


def _render_focus_table(
    per_class: dict, focus_classes: list[str], *, empty_message: str = "(none)"
) -> str:
    rows = [_class_row(name, per_class.get(name)) for name in focus_classes]
    return _render_table(
        ["class", "ap50", "ap50_95", "precision", "recall"],
        rows or [[empty_message, "N/A", "N/A", "N/A", "N/A"]],
    )


def _render_base_table(
    per_class: dict,
    base_classes: list[str],
    delta: dict | None,
) -> str:
    delta_ap50 = (delta or {}).get("delta_ap50", {})
    delta_ap50_95 = (delta or {}).get("delta_ap50_95", {})
    rows: list[list[str]] = []
    for name in base_classes:
        if name not in per_class:
            continue
        row = _class_row(name, per_class[name])
        row.extend(
            [
                _fmt_delta(delta_ap50.get(name)),
                _fmt_delta(delta_ap50_95.get(name)),
            ]
        )
        rows.append(row)
    if not rows and per_class:
        for name, metrics in per_class.items():
            row = _class_row(name, metrics)
            row.extend([_fmt_delta(delta_ap50.get(name)), _fmt_delta(delta_ap50_95.get(name))])
            rows.append(row)
    return _render_table(
        ["class", "ap50", "ap50_95", "precision", "recall", "Δap50", "Δap50_95"],
        rows or [["(none)", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]],
    )


def _render_summary_table(summary: dict, delta: dict | None) -> str:
    rows = [
        ["map50", _fmt(summary.get("map50"))],
        ["map50_95", _fmt(summary.get("map50_95"))],
        ["focus_map50", _fmt(summary.get("focus_map50"))],
        ["focus_map50_95", _fmt(summary.get("focus_map50_95"))],
        ["base_map50", _fmt(summary.get("base_map50"))],
        ["base_map50_95", _fmt(summary.get("base_map50_95"))],
    ]
    if delta is not None:
        rows.append(["delta_base_map50", _fmt_delta(delta.get("delta_base_map50"))])
    return _render_table(["metric", "value"], rows)


def _split_order(metrics_by_split: dict) -> list[str]:
    preferred = ["val", "test"]
    ordered = [s for s in preferred if s in metrics_by_split]
    ordered += [s for s in metrics_by_split if s not in ordered]
    return ordered


def _render_split_section(
    split: str,
    metrics: dict,
    *,
    delta: dict | None,
    focus_classes: list[str],
    base_classes: list[str],
    focus_empty_message: str,
) -> list[str]:
    return [
        f"## Split: {split}",
        "",
        "### Focus classes",
        "",
        _render_focus_table(
            metrics.get("per_class", {}), focus_classes, empty_message=focus_empty_message
        ),
        "",
        "### Base classes",
        "",
        _render_base_table(
            metrics.get("per_class", {}),
            base_classes,
            delta,
        ),
        "",
        "### Summary",
        "",
        _render_summary_table(metrics.get("summary", {}), delta),
    ]


def _render_report(
    meta: dict,
    metrics_by_split: dict[str, dict],
    *,
    baseline_val: dict | None,
    delta: dict | None,
    focus_classes: list[str],
    base_classes: list[str],
) -> str:
    wave = meta.get("wave", "unknown")
    lines = [
        f"# Eval report: {wave}",
        "",
        f"- **weights:** `{meta.get('weights', 'N/A')}`",
        f"- **data:** `{meta.get('data', 'N/A')}`",
    ]
    if meta.get("baseline"):
        lines.append(f"- **baseline:** `{meta['baseline']}`")

    focus_empty_message = _focus_empty_message(meta)

    for split in _split_order(metrics_by_split):
        lines.append("")
        lines.extend(
            _render_split_section(
                split,
                metrics_by_split[split],
                # Baseline delta (forgetting check) is only computed against
                # the val split; other splits (e.g. test) never get a delta.
                delta=delta if split == "val" else None,
                focus_classes=focus_classes,
                base_classes=base_classes,
                focus_empty_message=focus_empty_message,
            )
        )

    if baseline_val is not None:
        lines.extend(
            [
                "",
                "## Notes",
                "",
                "- Base-class forgetting uses **val** split vs baseline on the same data.yaml.",
                "- Δap50 negative on base classes indicates potential forgetting.",
            ]
        )
    return "\n".join(lines) + "\n"


def write_eval_run(
    out_dir: Path,
    *,
    meta: dict,
    metrics_by_split: dict[str, dict],
    baseline_val: dict | None,
    delta: dict | None,
    index_path: Path,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "run_meta.json", meta)

    for split, metrics in metrics_by_split.items():
        _write_json(out_dir / f"metrics_{split}.json", metrics)

    if baseline_val is not None:
        _write_json(out_dir / "metrics_baseline_val.json", baseline_val)
    if delta is not None:
        _write_json(out_dir / "baseline_delta.json", delta)

    metrics_val = metrics_by_split.get("val") or next(iter(metrics_by_split.values()))
    focus_classes, base_classes = _resolve_class_groups(meta)
    report = _render_report(
        meta,
        metrics_by_split,
        baseline_val=baseline_val,
        delta=delta,
        focus_classes=focus_classes,
        base_classes=base_classes,
    )
    (out_dir / "report.md").write_text(report)

    summary = metrics_val.get("summary", {})
    index_entry = {
        "wave": meta.get("wave"),
        "dir": str(out_dir.resolve()),
        "weights": meta.get("weights"),
        "map50_val": summary.get("map50"),
        "focus_map50": summary.get("focus_map50"),
        "delta_base_map50": delta.get("delta_base_map50") if delta else None,
        "ts": meta.get("ts") or datetime.now(timezone.utc).isoformat(),
    }
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(index_entry, ensure_ascii=False) + "\n")

    return out_dir
