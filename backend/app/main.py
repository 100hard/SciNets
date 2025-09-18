from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.db.database import test_postgres_connection
from app.services.storage import ensure_bucket_exists
from app.api.papers import router as papers_router
from app.db.migrate import apply_migrations
from app.db.pool import init_pool, close_pool


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as exc:  # Basic error handling for MVP
            print(f"Request error: {exc}")  # Log the actual error
            return JSONResponse(status_code=500, content={"detail": str(exc) if exc else "Unknown error"})

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.on_event("startup")
    async def on_startup():
        try:
            await test_postgres_connection()
        except Exception as exc:  # noqa: PERF203
            # Keep app up but log the error; for MVP we return 500 only on request
            print(f"Postgres connection check failed: {exc}")

        try:
            ensure_bucket_exists(settings.minio_bucket_papers)
        except Exception as exc:  # noqa: PERF203
            print(f"MinIO bucket ensure failed: {exc}")

        # Apply DB migrations and init pool
        try:
            print("Starting database migrations...")
            await apply_migrations()
            print("Migrations completed successfully")
            print("Initializing database pool...")
            await init_pool()
            print("Database pool initialized successfully")
        except Exception as exc:  # noqa: PERF203
            print(f"Startup DB init failed: {exc}")
            import traceback
            traceback.print_exc()

    @app.on_event("shutdown")
    async def on_shutdown():
        try:
            await close_pool()
        except Exception as exc:  # noqa: PERF203
            print(f"Shutdown DB pool close failed: {exc}")

    # Routers
    app.include_router(papers_router)

    return app


app = create_app()


