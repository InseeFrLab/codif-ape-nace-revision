import logging


def setup_logging():
    logging.basicConfig(
        filename="encode_ambiguous.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    )
