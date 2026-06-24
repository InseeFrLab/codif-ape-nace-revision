# config/logging.py
import logging
import sys

def setup_logging(log_file="encode_ambiguous.log"):
    """
    Configure logging pour logger à la fois dans un fichier et dans la console.
    """
    logger = logging.getLogger()  # Logger racine
    logger.setLevel(logging.DEBUG)  #logging.WARNING

    # --- Handler pour le fichier ---
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Tout log dans le fichier
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # --- Handler pour la console ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Tout log dans le terminal
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Évite que le logger racine duplique les messages si setup_logging est rappelé
    logger.propagate = False

    # for noisy_logger in (
    #     "botocore", "s3fs", "aiobotocore", "urllib3",
    #     "openai", "httpx", "httpcore",
    # ):
    #     logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return logger
