import logging

def setup_logging(logfile=None):
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile, mode='w'))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        handlers=handlers
    )
