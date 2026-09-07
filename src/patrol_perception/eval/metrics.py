from __future__ import annotations

from typing import Any


def empty_class_metrics() -> dict[str, float | None]:
    return {
        "ap50": None,
        "ap50_95": None,
        "precision": None,
        "recall": None,
    }


def build_metrics_doc(
    per_class: dict[str, dict],
    summary: dict,
    *,
    split: str,
    names: list[str],
) -> dict:
    return {
        "split": split,
        "names": list(names),
        "per_class": per_class,
        "summary": summary,
    }


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _as_list(value: Any) -> list | None:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _get_box(results: Any) -> Any:
    return getattr(results, "box", results)


def _per_class_from_box(box: Any, names: list[str]) -> dict[str, dict]:
    per_class: dict[str, dict] = {}
    ap_class_index = _as_list(getattr(box, "ap_class_index", None)) or None
    maps = getattr(box, "maps", None)
    ap50_values = _as_list(getattr(box, "ap50", None))
    p_values = _as_list(getattr(box, "p", None))
    r_values = _as_list(getattr(box, "r", None))

    for i, name in enumerate(names):
        metrics = empty_class_metrics()

        if ap_class_index:
            # Ultralytics only populates ap50/p/r/maps for classes present in
            # ap_class_index (i.e. classes with GT in this split). For absent
            # classes, `maps[i]` is filled with a sentinel (often the overall
            # mAP), NOT a per-class value, so ap50_95 must be gated the same
            # way as ap50/precision/recall rather than read unconditionally.
            if i not in ap_class_index:
                per_class[name] = metrics
                continue
            pos = ap_class_index.index(i)
            if ap50_values and pos < len(ap50_values):
                metrics["ap50"] = _coerce_float(ap50_values[pos])
            if p_values and pos < len(p_values):
                metrics["precision"] = _coerce_float(p_values[pos])
            if r_values and pos < len(r_values):
                metrics["recall"] = _coerce_float(r_values[pos])
            if maps is not None:
                try:
                    if len(maps) > i:
                        metrics["ap50_95"] = _coerce_float(maps[i])
                except TypeError:
                    pass
        else:
            if ap50_values and i < len(ap50_values):
                metrics["ap50"] = _coerce_float(ap50_values[i])
            if p_values and i < len(p_values):
                metrics["precision"] = _coerce_float(p_values[i])
            if r_values and i < len(r_values):
                metrics["recall"] = _coerce_float(r_values[i])
            if maps is not None:
                try:
                    if len(maps) > i:
                        metrics["ap50_95"] = _coerce_float(maps[i])
                except TypeError:
                    pass

        per_class[name] = metrics

    return per_class


def _summary_from_box(box: Any) -> dict[str, float]:
    summary: dict[str, float] = {}
    map50 = getattr(box, "map50", None)
    map50_95 = getattr(box, "map", None)
    if map50 is not None:
        summary["map50"] = _coerce_float(map50)
    if map50_95 is not None:
        summary["map50_95"] = _coerce_float(map50_95)
    return summary


def extract_from_ultralytics(
    results: Any,
    names: list[str],
    *,
    split: str = "val",
) -> dict:
    box = _get_box(results)
    per_class = _per_class_from_box(box, names)
    summary = _summary_from_box(box)
    return build_metrics_doc(
        per_class,
        summary,
        split=split,
        names=names,
    )
