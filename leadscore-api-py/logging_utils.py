# logging_utils.py
import logging
import sys
from pythonjsonlogger import jsonlogger

# Optional: cloud logging & error reporting clients imported lazily
def setup_logging(service_name="leadscore-api-py"):
    """
    Returns (logger, error_report_client_or_None).
    Logger writes structured JSON to stdout so Cloud Run / Cloud Logging stores it.
    """
    # JSON stdout logger (works well with Cloud Logging)
    logger = logging.getLogger(service_name)
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers during reloads
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
        json_formatter = jsonlogger.JsonFormatter(fmt)
        stream_handler.setFormatter(json_formatter)
        logger.addHandler(stream_handler)

    # Try to create an Error Reporting client if environment allows; fail silently otherwise.
    try:
        from google.cloud import errorreporting
        err_client = errorreporting.Client()
    except Exception:
        err_client = None

    return logger, err_client
