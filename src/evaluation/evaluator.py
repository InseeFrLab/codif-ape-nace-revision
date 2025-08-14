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

        # Step 2: Map prompts
        prompt_mapping = self.get_prompt_mapping(prompts, ground_truth)

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
        metrics = accuracies | {"eval_size": eval_df.shape[0], "mapping_ok": eval_df["mapping_ok"].sum()}
        return metrics, eval_df


    def get_prompt_mapping(self, prompts: List, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """
        Processes prompts and returns a DataFrame with liasse_numero, mapping_ok, and position.
        Make sure that prompt List and ground_truth dataFrame are similarly ordered.
        """

        pattern = r"'([\d]{2}\.[\d]{2}[A-Z])'"
        mapping = []
        ground_truth_c = ground_truth.copy().reset_index(drop=True)
        for idx, row in enumerate(ground_truth_c.to_dict(orient="records")):
            text = prompts[idx][1]["content"] # 1 to get "user prompt" dict, not "system prompt" dict

            # Retrieve the proposed code from the prompt
            proposed_codes = [c.replace(".", "") for c in re.findall(pattern, text)]

            manual_code = ground_truth_c.loc[idx, "apet_manual"]

            mapping_ok = manual_code in proposed_codes
            position = proposed_codes.index(manual_code) if mapping_ok else None

            mapping.append(
                {
                    "liasse_numero": ground_truth_c.loc[idx, "liasse_numero"],
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
