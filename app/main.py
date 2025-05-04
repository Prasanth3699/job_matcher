from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.session import engine
from app.db.base import Base
from app.core.config import get_settings
from app.api.router import api_router
from app.api.v1.endpoints.core import setup_instrumentation

settings = get_settings()


# ---------------------------------------------------------------------
# Lifespan handler replaces @app.on_event("startup"/"shutdown")
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------- startup -------------------------------------------
    Base.metadata.create_all(bind=engine)
    yield  # <--- application runs here
    # ------------- shutdown ------------------------------------------
    engine.dispose()


# ---------------------------------------------------------------------
# FastAPI app definition
# ---------------------------------------------------------------------
app = FastAPI(
    title="Resume Matcher API",
    description="API for matching resumes to job postings",
    version="1.0.0",
    lifespan=lifespan,  # <-- use the context manager
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# Prometheus / other instrumentation
setup_instrumentation(app)
