import re
from typing import Dict, List, Optional

import pandas as pd

from utils.data import get_ground_truth


class Evaluator:
    """
    Evaluator class to compute accuracy metrics
    based on ground truth and LLM results.
    """

    def evaluate(self, results: pd.DataFrame, prompts: pd.DataFrame) -> tuple[Dict, pd.DataFrame]:
        """
        Run the full evaluation pipeline.

        Args:
            results: DataFrame with model results.
            prompts: DataFrame with prompts used.
        Returns:
            - Dictionary with accuracy metrics.
            - DataFrame containing the merged evaluation data with ground truth, prompt mapping, and results.
        """
        # Step 1: Get ground truth and make sure it is a subset of results
        ground_truth = get_ground_truth()
        ground_truth = ground_truth[ground_truth["liasse_numero"].isin(results["liasse_numero"])]

        if ground_truth.empty:
            raise ValueError(
                "No annotated rows found in the results sample. "
                "Either run the pipeline with only_annotated=True so the sample is drawn from "
                "annotated rows only, or increase sample_size to improve overlap with ground truth."
            )

        # Step 2: Map prompts
        # `prompts` is index-aligned with `results` (and with `data`), but
        # `ground_truth` comes back from get_ground_truth() filtered by isin
        # — its row order is independent. Build a liasse→prompt-index map so
        # get_prompt_mapping can look up the right prompt for each ground-truth row.
        liasse_to_idx = {ln: i for i, ln in enumerate(results["liasse_numero"].tolist())}
        prompt_mapping = self.get_prompt_mapping(prompts, ground_truth, liasse_to_idx)

        # Step 3: Merge prompt mapping
        ground_truth = ground_truth.merge(prompt_mapping, on="liasse_numero", how="inner")

        # Step 4: Merge results with ground truth
        eval_df = ground_truth.merge(results[["liasse_numero", "nace2025", "codable"]], on="liasse_numero", how="inner")

        # Step 5: Compute accuracy metrics
        accuracies = (
            self.calculate_accuracy(eval_df)
            | self.calculate_accuracy(eval_df, filter_col="mapping_ok")
            | self.calculate_accuracy(eval_df, filter_col="codable")
        )

        # Step 6: Compute additional metrics
        metrics = (
            accuracies
            | {
                "eval_size": eval_df.shape[0],
                "mapping_ok": eval_df["mapping_ok"].sum(),
                "mapping_ok_pct": (eval_df["mapping_ok"].sum()) / eval_df.shape[0]
            }
        )
        return metrics, eval_df

    def get_prompt_mapping(
        self, prompts: List, ground_truth: pd.DataFrame, liasse_to_idx: Dict[str, int],
    ) -> pd.DataFrame:
        """
        For each ground-truth row, look up the prompt that was sent for that
        liasse and check whether the labelled NACE code appears in the prompt's
        proposed list. Returns a DataFrame with liasse_numero, mapping_ok, and
        position (rank of the labelled code in the proposed list, 0-based).
        """
        pattern = r"'([\d]{2}\.[\d]{2}[A-Z])'"
        mapping = []
        for row in ground_truth.to_dict(orient="records"):
            liasse = row["liasse_numero"]
            idx = liasse_to_idx.get(liasse)
            if idx is None:
                continue
            text = prompts[idx][1]["content"]  # 1 to get "user prompt" dict, not "system prompt" dict

            # Retrieve the proposed codes from the prompt
            proposed_codes = [c.replace(".", "") for c in re.findall(pattern, text)]

            manual_code = row["apet_manual"]

            mapping_ok = manual_code in proposed_codes
            position = proposed_codes.index(manual_code) if mapping_ok else None

            mapping.append(
                {
                    "liasse_numero": liasse,
                    "mapping_ok": mapping_ok,
                    "position": position,
                }
            )
        return pd.DataFrame(mapping)

    def calculate_accuracy(self, eval_df: pd.DataFrame, filter_col: Optional[str] = None) -> Dict:
        """
        Calculates accuracy at different levels.
        If `filter_col` is provided, only considers rows where `filter_col` is True.
        """
        filtered_df = eval_df if filter_col is None else eval_df[eval_df[filter_col]]

        return {
            f"accuracy_{filter_col or 'overall'}_lvl_{i}": round(
                (filtered_df["apet_manual"].str[:i] == filtered_df["nace2025"].str[:i]).mean() * 100, 2
            )
            for i in [5, 4, 3, 2, 1]
        }
