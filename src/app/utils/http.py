import httpx
from urllib3.util.retry import Retry
from httpx import Timeout

_client: httpx.AsyncClient | None = None

def get_async_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=5.0)
        )
    return _client