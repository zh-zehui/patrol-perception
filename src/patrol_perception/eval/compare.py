from pathlib import Path

import yaml


def load_eval_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def resolve_baseline(cli: str | None, cfg: dict, *, no_baseline: bool = False) -> str | None:
    """CLI --baseline overrides eval.yaml baseline.default; --no-baseline skips both."""
    if no_baseline:
        return None
    if cli:
        return cli
    default = (cfg.get("baseline") or {}).get("default")
    if not default:
        return None
    return str(default)


def summarize_metrics(per_class, *, focus, base):
    def mean_ap(names, key):
        vals = [
            per_class[n][key]
            for n in names
            if n in per_class and per_class[n].get(key) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    all_names = list(per_class)
    return {
        "map50": mean_ap(all_names, "ap50"),
        "map50_95": mean_ap(all_names, "ap50_95"),
        "focus_map50": mean_ap(focus, "ap50"),
        "focus_map50_95": mean_ap(focus, "ap50_95"),
        "base_map50": mean_ap(base, "ap50"),
        "base_map50_95": mean_ap(base, "ap50_95"),
    }


def compute_delta(current, baseline, *, base_classes):
    a = current["per_class"]
    b = baseline["per_class"]
    common = sorted(set(a) & set(b))
    delta_ap50 = {n: a[n]["ap50"] - b[n]["ap50"] for n in common}
    delta_ap50_95 = {n: a[n]["ap50_95"] - b[n]["ap50_95"] for n in common}
    base_common = [n for n in base_classes if n in delta_ap50]
    delta_base = (
        sum(delta_ap50[n] for n in base_common) / len(base_common)
        if base_common
        else None
    )
    return {
        "delta_ap50": delta_ap50,
        "delta_ap50_95": delta_ap50_95,
        "delta_base_map50": delta_base,
        "compared_classes": common,
    }
