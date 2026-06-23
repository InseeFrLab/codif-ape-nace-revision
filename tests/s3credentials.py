"""
Test S3 credentials stored in .env.
Verifies that AWS_* variables are set and that the mapping Excel file is readable.
"""

import os
import sys
import pandas as pd
import s3fs
from dotenv import load_dotenv

load_dotenv(override=True)

# --------------------------------------------------
# Check env variables
# --------------------------------------------------

_REQUIRED_ENV = ["AWS_S3_ENDPOINT", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
_missing = [v for v in _REQUIRED_ENV if not os.getenv(v)]
if _missing:
    sys.exit(f"Variables d'environnement manquantes : {', '.join(_missing)}")

URL_MAPPING_TABLE = "s3://projet-ape/NAF-revision/table-correspondance-naf2025.xls"

# --------------------------------------------------
# Tests
# --------------------------------------------------

def make_filesystem() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def test_s3_connection(fs: s3fs.S3FileSystem) -> None:
    """Checks that the bucket is reachable with the provided credentials."""
    assert fs.exists("projet-ape"), "Bucket 'projet-ape' inaccessible — vérifier les credentials"
    print("[OK] Connexion S3 établie")


def test_mapping_file_readable(fs: s3fs.S3FileSystem) -> pd.DataFrame:
    """Checks that the mapping Excel file can be opened and parsed."""
    with fs.open(URL_MAPPING_TABLE) as f:
        df = pd.read_excel(f, dtype=str)
    assert not df.empty, "Le fichier Excel est vide"
    print(f"[OK] Fichier lu : {URL_MAPPING_TABLE} ({len(df)} lignes, {len(df.columns)} colonnes)")
    return df


if __name__ == "__main__":
    fs = make_filesystem()
    test_s3_connection(fs)
    test_mapping_file_readable(fs)
