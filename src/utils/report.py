"""Markdown report generation for encoding runs (CAG / RAG).

The report is logged as an MLflow artifact alongside the run metrics."""

from datetime import datetime

from utils.nace import SECTION_TITLE, code_to_section


_NACE_LEVELS = {
    1: "Section",
    2: "Division",
    3: "Group",
    4: "Class",
    5: "Subclass",
}


def _accuracy_table(metrics: dict, family: str) -> str:
    rows = ["| Level | NACE granularity | Accuracy |", "|---|---|---|"]
    for lvl in range(1, 6):
        val = metrics.get(f"accuracy_{family}_lvl_{lvl}")
        val_str = f"{val:.2f}%" if isinstance(val, (int, float)) else "—"
        rows.append(f"| {lvl} | {_NACE_LEVELS[lvl]} | {val_str} |")
    return "\n".join(rows)


def _fmt_num(metrics: dict, key: str, *, decimals: int = 1) -> str:
    v = metrics.get(key)
    if v is None:
        return "—"
    return f"{v:.{decimals}f}" if isinstance(v, float) else str(v)


def _section_label(letter) -> str:
    """Render `letter — title` for a known section, fallback to `?` otherwise."""
    if letter is None or letter not in SECTION_TITLE:
        return "?"
    return f"{letter} — {SECTION_TITLE[letter]}"


def _per_section_error_table(df_eval) -> str:
    """Accuracy / error rate at full code (lvl 5) grouped by the truth's
    NACE section (letter A–U derived from the 2-digit division). Sorted by
    error rate descending so the worst sections come first."""
    if df_eval is None or df_eval.empty:
        return "_No evaluation data available._"

    section = df_eval["apet_manual"].apply(code_to_section)
    truth5 = df_eval["apet_manual"].astype(str).str[:5]
    pred5 = df_eval["nace2025"].astype(str).str[:5]
    correct_lvl5 = (truth5 == pred5)

    grouped = correct_lvl5.groupby(section, dropna=False).agg(["size", "sum"])
    grouped.columns = ["n", "n_correct"]
    grouped["accuracy_pct"] = grouped["n_correct"] / grouped["n"] * 100
    grouped["error_rate_pct"] = 100 - grouped["accuracy_pct"]
    grouped = grouped.sort_values("error_rate_pct", ascending=False)

    rows = [
        "| Section | N (truth) | Correct (lvl 5) | Accuracy | Error rate |",
        "|---|---|---|---|---|",
    ]
    for sec, r in grouped.iterrows():
        rows.append(
            f"| {_section_label(sec)} | {int(r['n'])} | {int(r['n_correct'])} | "
            f"{r['accuracy_pct']:.2f}% | {r['error_rate_pct']:.2f}% |"
        )
    return "\n".join(rows)


def _distribution_skew_table(df_eval) -> tuple[str, float]:
    """Compare the predicted section distribution against the labelled
    section distribution, on the subset of codable predictions (so both
    distributions are normalized over the same denominator). Returns
    (markdown_table, total_variation_distance)."""
    if df_eval is None or df_eval.empty:
        return "_No evaluation data available._", 0.0

    codable = df_eval[df_eval["codable"] == True]
    total = len(codable)
    if total == 0:
        return "_No codable predictions to compare._", 0.0

    section_truth = codable["apet_manual"].apply(code_to_section)
    section_pred = codable["nace2025"].apply(code_to_section)
    sections = sorted(
        set(section_truth.dropna().unique()) | set(section_pred.dropna().unique())
    )
    # Append "unknown" bucket at the end if any code couldn't be mapped.
    if section_truth.isna().any() or section_pred.isna().any():
        sections.append(None)

    rows = [
        "| Section | Truth N | Truth % | Pred N | Pred % | Δ (pred − truth) pp |",
        "|---|---|---|---|---|---|",
    ]
    tvd = 0.0
    for sec in sections:
        if sec is None:
            n_truth = int(section_truth.isna().sum())
            n_pred = int(section_pred.isna().sum())
        else:
            n_truth = int((section_truth == sec).sum())
            n_pred = int((section_pred == sec).sum())
        pct_truth = n_truth / total * 100
        pct_pred = n_pred / total * 100
        delta = pct_pred - pct_truth
        tvd += abs(delta) / 100
        rows.append(
            f"| {_section_label(sec)} | {n_truth} | {pct_truth:.2f}% | "
            f"{n_pred} | {pct_pred:.2f}% | {delta:+.2f} |"
        )

    return "\n".join(rows), tvd / 2


def build_report(
    strategy, llm_name, collection_name, top_k, sample_size, only_annotated, metrics, run_name,
    *, df_eval=None,
) -> str:
    """Render the encoding-run metrics as a Markdown report.

    The Retriever-quality section and the retriever-related configuration rows
    are emitted only when `strategy` exposes a Qdrant handle (RAG case)."""
    is_rag = hasattr(strategy, "db")
    strategy_name = "RAG" if is_rag else "CAG"

    lines = [
        f"# Encoding Quality Report — {strategy_name}",
        "",
        f"- **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Run name:** `{run_name}`" if run_name else "- **Run name:** _(none)_",
        "",
        "## Configuration",
        "",
        "| Param | Value |",
        "|---|---|",
        f"| Strategy | {strategy_name} |",
        f"| LLM model | `{llm_name}` |",
        f"| Thinking | `{strategy.thinking}` |",
        f"| Max new tokens | {strategy.sampling_params['max_tokens']} |",
        f"| Temperature | {strategy.sampling_params['temperature']} |",
        f"| Sample size (requested) | {sample_size if sample_size is not None else 'full dataset'} |",
        f"| Only annotated | {only_annotated} |",
    ]
    if is_rag:
        lines += [
            f"| Qdrant collection | `{collection_name}` |",
            f"| Embedding model | `{getattr(strategy.db, 'model_name', None)}` |",
            f"| Top-k | {top_k} |",
        ]

    lines += [
        "",
        "## Coverage",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Evaluation size | {metrics.get('eval_size', '—')} |",
        f"| Coded | {metrics.get('num_coded', '—')} |",
        f"| Not coded | {metrics.get('num_not_coded', '—')} |",
        f"| % not coded | {metrics.get('pct_not_coded', '—')}% |",
        "",
        "## Accuracy",
        "",
        "### All items",
        "",
        _accuracy_table(metrics, "overall"),
        "",
        "### Codable items only",
        "",
        _accuracy_table(metrics, "codable"),
        "",
    ]

    skew_table, tvd = _distribution_skew_table(df_eval)
    lines += [
        "## Per-section breakdown",
        "",
        "### Error rate by NACE section",
        "",
        "Sections are the standard NACE 2025 (NAF Rev. 2.1) sections A–U, derived from the 2-digit division of each code. Sorted by error rate descending. \"Error rate\" measures cases where the LLM's full lvl-5 code differs from the labelled code, computed within each ground-truth section.",
        "",
        _per_section_error_table(df_eval),
        "",
        "### Distribution skew (predictions vs labels)",
        "",
        "Truth and prediction distributions are computed on the codable subset (non-codable predictions excluded). Δ in **percentage points** (positive = section over-predicted vs labels).",
        "",
        skew_table,
        "",
        f"**Total variation distance:** {tvd:.4f} _(0 = identical distributions, 1 = disjoint)_",
        "",
    ]

    if is_rag:
        mapping_ok_pct = metrics.get("mapping_ok_pct")
        mapping_ok_pct_str = f"{mapping_ok_pct * 100:.2f}%" if isinstance(mapping_ok_pct, (int, float)) else "—"
        lines += [
            "## Retriever quality",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Target code present in top-k | {metrics.get('mapping_ok', '—')} / {metrics.get('eval_size', '—')} |",
            f"| Top-k recall | {mapping_ok_pct_str} |",
            "",
            "### Accuracy on items where target code is in top-k",
            "",
            _accuracy_table(metrics, "mapping_ok"),
            "",
        ]

    lines += [
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval time | {_fmt_num(metrics, 'retrieval_time_mn')} min |",
        f"| Generation time | {_fmt_num(metrics, 'generation_time_mn')} min |",
        f"| Generation throughput | {_fmt_num(metrics, 'generation_iter_per_sec', decimals=2)} iter/s |",
        f"| Completion tokens (mean / max / min) | {_fmt_num(metrics, 'completion_tokens_mean')} / {_fmt_num(metrics, 'completion_tokens_max', decimals=0)} / {_fmt_num(metrics, 'completion_tokens_min', decimals=0)} |",
        f"| Prompt tokens (mean / max / min) | {_fmt_num(metrics, 'prompt_tokens_mean')} / {_fmt_num(metrics, 'prompt_tokens_max', decimals=0)} / {_fmt_num(metrics, 'prompt_tokens_min', decimals=0)} |",
        "",
    ]

    return "\n".join(lines)
