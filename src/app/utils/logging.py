import logging
import os
from contextvars import ContextVar
from src.app.middleware.request_id import request_id_ctx

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_ctx.get()
        record.request_id = rid if rid is not None else "-"
        return True

def setup_logging(level: str | int | None = None) -> None:
    """
    Configure Python logging so each log line has request_id.
    Cloud Run ingests stdout/stderr and surfaces JSON automatically if you use json logs.
    Here we keep it simple, but you can swap the Formatter for a JSON one later.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        # avoid double init in hot-reload
        return

    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s [rid=%(request_id)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    ))

    root.addHandler(handler)
