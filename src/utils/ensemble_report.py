"""Markdown report for the ensemble evaluation (`4_evaluate_strategies.py`).

Summarises individual-model and majority-voting accuracies at each NACE level,
plus agreement statistics across the runs."""

from datetime import datetime
from typing import Dict, List


_NACE_LEVELS = {
    1: "Section",
    2: "Division",
    3: "Group",
    4: "Class",
    5: "Subclass",
}


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}%" if isinstance(v, (int, float)) else str(v)


def _accuracy_row(
    model_key: str, accuracies: Dict[str, float], levels: List[int]
) -> str:
    """Build one Markdown table row of `accuracy_{model_key}_lvl_{n}` values."""
    cells = [f"`{model_key}`"]
    for lvl in levels:
        cells.append(_fmt_pct(accuracies.get(f"accuracy_{model_key}_lvl_{lvl}")))
    return "| " + " | ".join(cells) + " |"


def _accuracy_table(
    model_keys: List[str],
    accuracies: Dict[str, float],
    levels: List[int],
) -> str:
    header_cells = ["Model"] + [f"Lvl {l} ({_NACE_LEVELS[l]})" for l in levels]
    sep = ["---"] * len(header_cells)
    rows = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    rows.extend(_accuracy_row(k, accuracies, levels) for k in model_keys)
    return "\n".join(rows)


def _runs_table(models: Dict[str, Dict]) -> str:
    rows = [
        "| Model | Run ID | Output path |",
        "| --- | --- | --- |",
    ]
    for name, cfg in models.items():
        rows.append(f"| `{name}` | `{cfg['run_id']}` | `{cfg['path']}` |")
    return "\n".join(rows)


def _agreement_summary(agreement: Dict[str, float]) -> str:
    rows = [
        "| Metric | Count | Share |",
        "| --- | --- | --- |",
    ]
    total = agreement.get("total_samples", 0)
    for key, label in [
        ("full_agreement", "Full agreement (all models, same code)"),
        ("partial_agreement", "Partial agreement (all non-null, same code, some missing)"),
        ("all_different", "All different (every model gave a distinct code)"),
        ("all_none", "All missing (no model produced a code)"),
    ]:
        count = agreement.get(f"{key}_count", 0)
        pct = agreement.get(f"{key}_pct", 0.0)
        rows.append(f"| {label} | {count} / {total} | {pct:.2f}% |")
    return "\n".join(rows)


def _pairwise_agreement_table(agreement: Dict[str, float]) -> str:
    rows = [
        "| Model A | Model B | Agreement |",
        "| --- | --- | --- |",
    ]
    for key, value in agreement.items():
        if not key.startswith("agreement_") or "_vs_" not in key:
            continue
        body = key[len("agreement_"):]
        a, b = body.split("_vs_", 1)
        rows.append(f"| `{a}` | `{b}` | {value:.2f}% |")
    return "\n".join(rows) if len(rows) > 2 else "_No pairwise agreement available._"


def build_ensemble_report(
    *,
    models: Dict[str, Dict],
    accuracies: Dict[str, Dict[str, float]],
    agreement: Dict[str, float],
    levels: List[int],
    ensemble_methods: List[str],
    eval_size: int,
    final_output_path: str,
    export_final: bool,
) -> str:
    """Render a Markdown report summarising the ensemble evaluation."""
    base_keys = list(models.keys())
    all_keys = base_keys + list(ensemble_methods)

    lines = [
        "# Ensemble Evaluation Report — Majority Voting",
        "",
        f"- **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Evaluation size (after ground-truth merge):** {eval_size}",
        f"- **Ensemble strategy kept:** majority voting (ties broken by first-model order)",
        "",
        "## Runs",
        "",
        _runs_table(models),
        "",
        "## Accuracy — raw (no filter)",
        "",
        "Compares each model's prediction (and the voting ensemble) against the manual code at each NACE granularity level.",
        "",
        _accuracy_table(all_keys, accuracies["raw"], levels),
        "",
        "## Accuracy — codable subset (per-model only)",
        "",
        "Restricted to rows where the model flagged the prediction as `codable=True`. The ensemble does not have an analogous filter (it's a vote across all rows), so only individual models appear here.",
        "",
        _accuracy_table(base_keys, accuracies["codable"], levels),
        "",
        "## Accuracy — `mapping_ok` subset",
        "",
        "Restricted to rows where the manual NACE 2025 code is one of the candidates produced by the NAF2008→NAF2025 mapping (i.e. cases where the ground-truth answer was actually reachable).",
        "",
        _accuracy_table(all_keys, accuracies["mapping_ok"], levels),
        "",
        "## Model agreement",
        "",
        _agreement_summary(agreement),
        "",
        "### Pairwise agreement (full-code identity rate)",
        "",
        _pairwise_agreement_table(agreement),
        "",
        "## Final export",
        "",
        f"- **Target path:** `{final_output_path}`",
        f"- **Exported:** {'yes' if export_final else 'no (EXPORT_FINAL=False)'}",
        "",
    ]
    return "\n".join(lines)
