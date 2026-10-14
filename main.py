# main.py
import logging
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.app.routers.validate import router as validate_router
from src.app.middleware.request_id import request_id_middleware
from src.app.utils.http import get_async_client
from src.app.utils.logging import setup_logging

log = logging.getLogger("uvicorn.error")

def create_app() -> FastAPI:
    setup_logging()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        get_async_client()
        logging.getLogger(__name__).info("Startup complete")
        yield
        # Shutdown
        client = get_async_client()
        await client.aclose()
        logging.getLogger(__name__).info("Shutdown complete")

    app = FastAPI(title="VLM Validator Service",lifespan=lifespan)

    app.middleware("http")(request_id_middleware(app))
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],            # tighten in prod
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(validate_router)

    @app.get("/healthz", response_class=PlainTextResponse)
    async def healthz() -> str:
        return {"status": "ok"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
