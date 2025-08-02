from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials

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
    
    # Initialize services
    from app.services.rabbitmq_client import get_rabbitmq_client
    try:
        # Pre-initialize RabbitMQ client to test connection
        await get_rabbitmq_client()
        print("RabbitMQ client initialized successfully")
    except Exception as e:
        print(f"Warning: RabbitMQ client initialization failed: {e}")
    
    yield  # <--- application runs here
    
    # ------------- shutdown ------------------------------------------
    from app.services.rabbitmq_client import cleanup_rabbitmq_client
    await cleanup_rabbitmq_client()
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
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Middleware
# app.add_middleware(MLSubscriptionMiddleware)

# Routers
app.include_router(api_router, prefix=settings.API_V1_STR)

# Prometheus / other instrumentation
setup_instrumentation(app)


app.middleware("http")
