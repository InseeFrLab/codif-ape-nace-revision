"""Markdown reports showing concrete examples of pipeline errors.

Three independent reports are produced (each capped at `max_examples`, with
graceful handling when fewer cases are available):

- `build_retriever_errors_report`: target NACE code missing from the proposed
  list (retriever miss for RAG, NACE08→NACE2025 mapping miss for CAG).
- `build_llm_errors_report`: target NACE code was in the proposed list but the
  LLM picked another code.
- `build_not_codable_report`: LLM deemed the description not codable. Only
  annotated cases are shown so that the labelled NACE code can be displayed.
"""

from datetime import datetime
from typing import Dict, List, Optional


def _get_user_prompt(prompts: List[List[Dict]], idx: Optional[int]) -> str:
    """Return the user-message content at the given prompt index.

    The evaluator already relies on the same `prompts[idx][1]` convention,
    so any deviation here would be a pipeline-wide bug, not a local issue."""
    if idx is None:
        return "_(prompt index not found)_"
    try:
        return prompts[idx][1]["content"]
    except (IndexError, KeyError, TypeError):
        return "_(prompt unavailable)_"


def _normalize_code(s) -> str:
    return str(s).replace(".", "") if s is not None else ""


def _is_nan(v) -> bool:
    try:
        return v != v
    except Exception:
        return False


def _build_liasse_idx(results) -> Dict:
    """Map liasse_numero → 0-based position in `prompts` (results is index-aligned)."""
    return {ln: i for i, ln in enumerate(results["liasse_numero"].tolist())}


def _header(
    title: str, strategy_name: str, total: int, max_examples: int, run_name: Optional[str],
) -> List[str]:
    showing = min(total, max_examples) if total > 0 else 0
    return [
        f"# {title} — {strategy_name}",
        "",
        f"- **Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Run name:** `{run_name}`" if run_name else "- **Run name:** _(none)_",
        f"- **Max examples:** {max_examples}",
        f"- **Total matching cases:** {total} (showing {showing})",
        "",
        "---",
        "",
    ]


def build_retriever_errors_report(
    strategy, prompts, results, df_eval, max_examples: int = 5, run_name: Optional[str] = None,
) -> str:
    is_rag = hasattr(strategy, "db")
    strategy_name = "RAG" if is_rag else "CAG"
    title = (
        "Retriever errors — target NACE code missing from the top-k"
        if is_rag
        else "Coverage errors — target NACE code missing from the proposed list"
    )

    missing = df_eval[df_eval["mapping_ok"] == False]
    liasse_to_idx = _build_liasse_idx(results)
    lines = _header(title, strategy_name, len(missing), max_examples, run_name)

    if len(missing) == 0:
        lines.append("_No matching cases in this run._")
        return "\n".join(lines)

    for i, row in enumerate(missing.head(max_examples).itertuples(), start=1):
        prompt_idx = liasse_to_idx.get(row.liasse_numero)
        lines += [
            f"## Example {i}",
            "",
            f"- **liasse_numero:** `{row.liasse_numero}`",
            f"- **Target NACE 2025 (manual):** `{row.apet_manual}`",
            f"- **LLM choice:** `{row.nace2025}` (codable={row.codable})",
            "",
            "```",
            _get_user_prompt(prompts, prompt_idx),
            "```",
            "",
        ]

    return "\n".join(lines)


def build_llm_errors_report(
    strategy, prompts, results, df_eval, max_examples: int = 5, run_name: Optional[str] = None,
) -> str:
    is_rag = hasattr(strategy, "db")
    strategy_name = "RAG" if is_rag else "CAG"

    target_norm = df_eval["apet_manual"].apply(_normalize_code)
    chosen_norm = df_eval["nace2025"].apply(_normalize_code)
    wrong = df_eval[
        (df_eval["mapping_ok"] == True)
        & (df_eval["codable"] == True)
        & (target_norm != chosen_norm)
    ]
    liasse_to_idx = _build_liasse_idx(results)
    lines = _header(
        "LLM errors — wrong choice despite the target being in the proposed list",
        strategy_name, len(wrong), max_examples, run_name,
    )

    if len(wrong) == 0:
        lines.append("_No matching cases in this run._")
        return "\n".join(lines)

    for i, row in enumerate(wrong.head(max_examples).itertuples(), start=1):
        prompt_idx = liasse_to_idx.get(row.liasse_numero)
        position_str = (
            f" (rank in proposed list: {int(row.position) + 1})"
            if getattr(row, "position", None) is not None and not _is_nan(row.position)
            else ""
        )
        lines += [
            f"## Example {i}",
            "",
            f"- **liasse_numero:** `{row.liasse_numero}`",
            f"- **Target NACE 2025 (manual):** `{row.apet_manual}`{position_str}",
            f"- **LLM choice:** `{row.nace2025}`",
            "",
            "```",
            _get_user_prompt(prompts, prompt_idx),
            "```",
            "",
        ]

    return "\n".join(lines)


def build_not_codable_report(
    strategy, prompts, results, df_eval, max_examples: int = 5, run_name: Optional[str] = None,
) -> str:
    """Pull from `df_eval` rather than `results` so that the labelled NACE
    code (`apet_manual`) is available — only annotated cases are shown."""
    is_rag = hasattr(strategy, "db")
    strategy_name = "RAG" if is_rag else "CAG"

    not_codable = df_eval[df_eval["codable"] == False]
    liasse_to_idx = _build_liasse_idx(results)
    lines = _header(
        "Not-codable — LLM deemed the description insufficient to assign a NACE code",
        strategy_name, len(not_codable), max_examples, run_name,
    )

    if len(not_codable) == 0:
        lines.append("_No matching cases in this run._")
        return "\n".join(lines)

    for i, row in enumerate(not_codable.head(max_examples).itertuples(), start=1):
        prompt_idx = liasse_to_idx.get(row.liasse_numero)
        lines += [
            f"## Example {i}",
            "",
            f"- **liasse_numero:** `{row.liasse_numero}`",
            f"- **Labelled NACE 2025 (manual):** `{row.apet_manual}`",
            "",
            "```",
            _get_user_prompt(prompts, prompt_idx),
            "```",
            "",
        ]

    return "\n".join(lines)
