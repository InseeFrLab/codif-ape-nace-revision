import duckdb
import json
import pandas as pd
from utils.data import get_file_system
fs = get_file_system()

conn = duckdb.connect()
s3_path = "s3://projet-ape/extractions/20241027_sirene4.parquet"
nature_path = "s3://projet-ape/data/nature.json"
cj_path = "s3://projet-ape/data/cj.json"

with fs.open(nature_path) as f:
    nature_data = json.load(f)
nature_df = conn.from_df(pd.DataFrame(nature_data))

with fs.open(cj_path) as f:
    cj_data = json.load(f)

cj_df = conn.from_df(pd.DataFrame(cj_data))

query = f"""
    COPY (
        SELECT
            s.*,
            n.* EXCLUDE(code) RENAME (
                libelle AS lib_nat
            ),
            c.* EXCLUDE(code) RENAME (
                libelle AS lib_cj
            ),
            CASE
                WHEN s.activ_nat_et = '99' THEN s.activ_nat_lib_et
                ELSE n.libelle
            END AS activ_nat_lib_et
        FROM
            read_parquet('{s3_path}') s
        LEFT JOIN
            nature_df n ON s.activ_nat_et = n.code
        LEFT JOIN
            cj_df c ON s.cj = c.code
    ) TO 's3://projet-ape/extractions/20250825_sirene4.parquet' (FORMAT PARQUET)
"""

conn.execute(query)

# result.columns


# result[["activ_nat_et", "lib_nat"]].drop_duplicates().sort_values("activ_nat_et")
# result[["cj", "lib_cj"]].drop_duplicates().sort_values("cj")

# result[["activ_nat_et", "lib_nat", "activ_nat_lib_et", "activ_nat_lib_et2"]] \
#     .drop_duplicates() \
#     .sort_values("activ_nat_et") \
#     .query("activ_nat_et != '99'") \
#     .reset_index()
