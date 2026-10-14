import uuid
import time
from typing import Callable
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response
from contextvars import ContextVar

# Expose a ContextVar so anything (validators, http client) can read the current request id
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

def _new_request_id() -> str:
    return str(uuid.uuid4())

def request_id_middleware(app: ASGIApp) -> Callable:
    async def middleware(request: Request, call_next):
        # 1) read incoming id or create a new one
        rid = request.headers.get("x-request-id") or _new_request_id()
        token = request_id_ctx.set(rid)   # bind to context for downstream code
        request.state.request_id = rid

        # 2) timing (optional but very useful)
        t0 = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            dt_ms = int((time.perf_counter() - t0) * 1000)

        # 3) always return x-request-id header and basic timing
        response.headers["x-request-id"] = rid
        response.headers["server-timing"] = f"total;dur={dt_ms}"

        # 4) cleanup
        request_id_ctx.reset(token)
        return response
    return middleware
