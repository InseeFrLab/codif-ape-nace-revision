from dotenv import load_dotenv

from .langfuse import setup_langfuse
from .logging import setup_logging


def setup():
    """Global setup routine"""
    # Load variables from .env into os.environ
    load_dotenv(override=True)
    # Setup logging
    setup_logging()
    # Setup Langfuse
    setup_langfuse()
