import logging
import os
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
import s3fs

from constants.data import VAR_TO_KEEP
from constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_GROUND_TRUTH,
    URL_MAPPING_TABLE,
    URL_PROMPTS_RAG,
    URL_SIRENE4_EXTRACTION,
)
from mappings.mappings import get_mapping


def get_file_system(token=None) -> s3fs.S3FileSystem:
    """
    Creates and returns an S3 file system instance using the s3fs library.

    This function configures the S3 file system with endpoint URL and credentials
    obtained from environment variables, enabling interactions with the specified
    S3-compatible storage. Optionally, a security token can be provided for session-based
    authentication.

    Parameters:
    -----------
    token : str, optional
        A temporary security token for session-based authentication. This is optional and
        should be provided when using session-based credentials.

    Returns:
    --------
    s3fs.S3FileSystem
        An instance of the S3 file system configured with the specified endpoint and
        credentials, ready to interact with S3-compatible storage.

    Environment Variables:
    ----------------------
    AWS_S3_ENDPOINT : str
        The S3 endpoint URL for the storage provider, typically in the format `https://{endpoint}`.
    AWS_ACCESS_KEY_ID : str
        The access key ID for authentication.
    AWS_SECRET_ACCESS_KEY : str
        The secret access key for authentication.

    Example:
    --------
    fs = get_file_system(token="your_temporary_token")
    """

    options = {
        "client_kwargs": {"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        "key": os.environ["AWS_ACCESS_KEY_ID"],
        "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    }

    if token is not None:
        options["token"] = token

    return s3fs.S3FileSystem(**options)


def merge_dataframes(df_dict: dict, merge_on, columns_to_rename=None, how="inner"):
    """
    Merge a dictionary of pandas DataFrames.

    Parameters:
    -----------
    df_dict : dict
        Dictionary of pandas DataFrames to merge with their names
    merge_on : str or list
        Column(s) to merge on
    columns_to_rename : dict, optional
        Dictionary specifying which columns to rename with suffix for each DataFrame
        Example: {"nace2025": "nace2025_{key}", "codable": "codable_{key}"}
    how : str, default 'inner'
        Type of merge to be performed: 'left', 'right', 'outer', 'inner'

    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame
    """
    if not df_dict:
        raise ValueError("DataFrame dictionary is empty")

    # Create a copy of the dictionary to avoid modifying the original
    processed_dfs = {}

    # Process each DataFrame: select columns and rename as needed
    for key, df in df_dict.items():
        # Select columns to keep
        temp_df = df[VAR_TO_KEEP].copy()

        # Rename columns if specified
        if columns_to_rename:
            rename_map = {col: pattern.format(key=key) for col, pattern in columns_to_rename.items()}
            temp_df.rename(columns=rename_map, inplace=True)

        processed_dfs[key] = temp_df

    # Start with the first DataFrame
    first_key = next(iter(processed_dfs))
    result = processed_dfs[first_key]

    # Merge with remaining DataFrames
    for key in list(processed_dfs.keys())[1:]:
        result = pd.merge(
            result,
            processed_dfs[key],
            on=merge_on,
            how=how,
        )

    return result


def load_excel_from_fs(fs, file_path):
    """Load an Excel file from the file system."""
    try:
        with fs.open(file_path) as f:
            return pd.read_excel(f, dtype=str)
    except Exception as e:
        logging.error(f"Failed to load file {file_path}: {e}")
        raise


def load_data_from_s3(query: str) -> pd.DataFrame:
    """Load data from S3 using DuckDB."""
    with duckdb.connect(database=":memory:") as con:
        try:
            con.execute(f"""
                SET s3_endpoint='{os.getenv("AWS_S3_ENDPOINT")}';
                SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
                SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
                SET s3_session_token='';
            """)
            result_df = con.execute(query).fetch_df()
            return result_df
        except Exception as e:
            logging.error(f"Failed to load data from S3: {e}")
            raise


def process_subset(data: pd.DataFrame, third: Optional[int]) -> pd.DataFrame:
    """Process only a subset of the data based on the 'third' argument."""
    if third is None:
        return data

    subset_size = len(data) // 3
    start_idx = subset_size * (third - 1)
    end_idx = subset_size * third if third != 3 else len(data)
    return data.iloc[start_idx:end_idx].reset_index(drop=True)


def fetch_mapping() -> Any:
    fs = get_file_system()
    # Load mapping data
    try:
        table_corres = load_excel_from_fs(fs, URL_MAPPING_TABLE)
        notes_ex = load_excel_from_fs(fs, URL_EXPLANATORY_NOTES)
        mapping = get_mapping(notes_ex, table_corres)
    except Exception as e:
        raise RuntimeError(f"Error loading mapping data: {e}")

    # Identify ambiguous mappings
    mapping_ambiguous = [code for code in mapping if len(code.naf2025) > 1]

    if not mapping_ambiguous:
        raise ValueError("No ambiguous codes found in mapping.")

    return mapping_ambiguous


def get_ambiguous_data(mapping: Any, third: bool, only_annotated: bool = False) -> pd.DataFrame:
    """
    Loads and processes data from multiple sources.

    Args:
        third (bool): Additional processing flag.
        only_annotated (bool): Flag to filter only annotated data.

    Returns:
        pd.DataFrame: Processed subset of data.
    """
    # Construct SQL query
    filter_columns_sql = ", ".join([v for v in VAR_TO_KEEP if v not in {"liasse_numero", "apet_finale"}])
    selected_columns_sql = ", ".join(VAR_TO_KEEP)
    ambiguous_codes = "', '".join([m.code.replace(".", "") for m in mapping])

    # Filter only annotated data if specified
    ground_truth_filter = (
        f"AND liasse_numero IN (SELECT liasse_numero FROM read_parquet('{URL_GROUND_TRUTH}'))" if only_annotated else ""
    )

    query = f"""
        WITH filtered_data AS (
            SELECT DISTINCT ON (liasse_numero) *
            FROM read_parquet('{URL_SIRENE4_EXTRACTION}')
            WHERE apet_finale IN ('{ambiguous_codes}')
            {ground_truth_filter}
        ),
        deduplicated_data AS (
            SELECT DISTINCT ON ({filter_columns_sql}) *
            FROM filtered_data
        )
        SELECT {selected_columns_sql}
        FROM deduplicated_data
        ORDER BY liasse_numero;
    """

    try:
        data = load_data_from_s3(query)
    except Exception as e:
        raise RuntimeError(f"Error loading data from S3: {e}")

    # Process data subset
    return process_subset(data, third)


def get_ground_truth() -> pd.DataFrame:
    """
    Retrieves and loads the ground truth data from a Parquet file.

    Returns:
        pd.DataFrame: A DataFrame with distinct liasse_numero, apet_manual, and NAF2008_code.
    """

    query = f"""
        SELECT DISTINCT ON (liasse_numero)
            liasse_numero,
            apet_manual,
            NAF2008_code
        FROM read_parquet('{URL_GROUND_TRUTH}')
        ORDER BY liasse_numero;
    """

    try:
        return load_data_from_s3(query)
    except Exception as e:
        raise RuntimeError(f"Error loading ground truth data: {e}")


def prompts_to_df(prompts: List[List[Dict]]) -> pd.DataFrame:
    rows = []
    for conversation in prompts:
        row = {}
        for message in conversation:
            role = message["role"]
            row[f"{role}_content"] = message["content"]
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_prompts(df: pd.DataFrame) -> List[List[Dict]]:
    prompt_list = []
    for _, row in df.iterrows():
        conversation = []
        for col in df.columns:
            role = col.replace("_content", "")
            content = row[col]
            conversation.append({"role": role, "content": content})
        prompt_list.append(conversation)
    return prompt_list


def save_prompts(prompts: List[List[Dict]]):
    fs = get_file_system()
    prompts_df = prompts_to_df(prompts)
    prompts_df.to_parquet(URL_PROMPTS_RAG.format(collection=os.getenv("COLLECTION_NAME")), filesystem=fs)


def load_prompts() -> List[List[Dict]]:
    fs = get_file_system()
    prompts_df = pd.read_parquet(URL_PROMPTS_RAG.format(collection=os.getenv("COLLECTION_NAME")), filesystem=fs)
    prompts = df_to_prompts(prompts_df)
    return prompts
