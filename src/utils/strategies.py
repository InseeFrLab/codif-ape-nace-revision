from collections import Counter

import pandas as pd


def select_labels_cascade(df: pd.DataFrame, model_columns: list, default_value=None) -> pd.Series:
    """
    Strategy 1: Cascade through models in order, taking first non-None value

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model predictions
    model_columns : list
        Ordered list of column names containing model predictions
    default_value : any, optional
        Value to use if all predictions are None

    Returns:
    --------
    pandas.Series
        Series containing selected labels
    """

    def cascade_select(row):
        for col in model_columns:
            if pd.notna(row[col]):
                return row[col]
        return default_value

    return df.apply(cascade_select, axis=1)


def select_labels_voting(df: pd.DataFrame, model_columns: list, default_value=None) -> pd.Series:
    """
    Strategy 2: Voting system with tiebreaker based on model priority

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model predictions
    model_columns : list
        Ordered list of column names containing model predictions
        (order determines priority for tiebreaking)
    default_value : any, optional
        Value to use if all predictions are None

    Returns:
    --------
    pandas.Series
        Series containing selected labels
    """

    def voting_select(row):
        # Get non-None predictions
        valid_predictions = [pred for pred in row[model_columns] if pd.notna(pred)]

        # If all are None, return default value
        if not valid_predictions:
            return default_value

        # If only one non-None prediction exists, use it
        if len(valid_predictions) == 1:
            return valid_predictions[0]

        # Count occurrences of each prediction
        prediction_counts = Counter(valid_predictions)

        # Find the prediction(s) with maximum votes
        max_votes = max(prediction_counts.values())
        top_predictions = [pred for pred, count in prediction_counts.items() if count == max_votes]

        # If there's only one top prediction, use it
        if len(top_predictions) == 1:
            return top_predictions[0]

        # In case of tie, use the highest priority model's prediction
        for col in model_columns:
            if pd.notna(row[col]) and row[col] in top_predictions:
                return row[col]

        # Fallback (should never reach here if input is valid)
        return default_value

    return df.apply(voting_select, axis=1)


def select_labels_weighted_voting(
    df: pd.DataFrame, model_columns: list, weights: dict = None, default_value=None
) -> pd.Series:
    """
    Weighted voting system where each model's vote has a different weight

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing model predictions
    model_columns : list
        Ordered list of column names containing model predictions
    weights : dict, optional
        Dictionary mapping model columns to their weights
        If None, equal weights are used
    """
    if weights is None:
        weights = {col: 1 for col in model_columns}

    def weighted_vote(row):
        # Get non-None predictions with their weights
        valid_predictions = {
            pred: weights[col] for col, pred in row[model_columns].items() if pd.notna(pred)
        }

        if not valid_predictions:
            return default_value

        # Sum weights for each unique prediction
        prediction_weights = {}
        for pred, weight in valid_predictions.items():
            prediction_weights[pred] = prediction_weights.get(pred, 0) + weight

        # Get prediction(s) with maximum weighted votes
        max_weight = max(prediction_weights.values())
        top_predictions = [
            pred for pred, weight in prediction_weights.items() if weight == max_weight
        ]

        return top_predictions[0] if len(top_predictions) == 1 else row[model_columns[0]]

    return df.apply(weighted_vote, axis=1)


def get_model_agreement_stats(df: pd.DataFrame, model_columns: list) -> dict:
    """
    Calculate comprehensive agreement statistics between models

    Returns:
    --------
    dict
        Dictionary containing various agreement statistics
    """
    total_rows = len(df)

    def get_row_agreement(row):
        predictions = [pred for pred in row[model_columns] if pd.notna(pred)]
        unique_predictions = set(predictions)

        return {
            "valid_predictions": len(predictions),
            "unique_predictions": len(unique_predictions),
            "full_agreement": len(unique_predictions) == 1
            and len(predictions) == len(model_columns),
            "partial_agreement": len(unique_predictions) == 1
            and len(predictions) < len(model_columns),
            "all_different": len(unique_predictions) == len(predictions) and len(predictions) > 1,
            "all_none": len(predictions) == 0,
        }

    # Calculate agreement stats for each row
    agreement_stats = df.apply(get_row_agreement, axis=1)

    # Aggregate statistics
    stats = {
        "total_samples": total_rows,
        "full_agreement_count": sum(row["full_agreement"] for row in agreement_stats),
        "partial_agreement_count": sum(row["partial_agreement"] for row in agreement_stats),
        "all_different_count": sum(row["all_different"] for row in agreement_stats),
        "all_none_count": sum(row["all_none"] for row in agreement_stats),
    }

    # Calculate percentages
    stats.update(
        {
            "full_agreement_pct": (stats["full_agreement_count"] / total_rows) * 100,
            "partial_agreement_pct": (stats["partial_agreement_count"] / total_rows) * 100,
            "all_different_pct": (stats["all_different_count"] / total_rows) * 100,
            "all_none_pct": (stats["all_none_count"] / total_rows) * 100,
        }
    )

    # Calculate pairwise agreement
    for i, model1 in enumerate(model_columns):
        for j, model2 in enumerate(model_columns[i + 1 :], i + 1):
            agreement = (df[model1] == df[model2]).mean() * 100
            stats[f"agreement_{model1}_vs_{model2}"] = agreement

    return stats
