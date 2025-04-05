from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.session import engine
from app.db.base import Base
from app.core.config import get_settings
from app.api.router import api_router
from app.api.v1.endpoints.core import setup_instrumentation


settings = get_settings()

app = FastAPI(
    title="Resume Matcher API",
    description="API for matching resumes to job postings",
    version="1.0.0",
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.on_event("startup")
def startup_event():
    """Initialize database tables on startup"""
    Base.metadata.create_all(bind=engine)


@app.on_event("shutdown")
def shutdown_event():
    """Clean up connections on shutdown"""
    engine.dispose()


# Setup Prometheus instrumentation
setup_instrumentation(app)
