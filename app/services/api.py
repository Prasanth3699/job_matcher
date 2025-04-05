import json
import uuid
from collections import defaultdict
import time

import prometheus_client
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBearer
from .api_models import APIErrorResponse
from app.utils.logger import logger


# Security
security = HTTPBearer()

# Rate limiting store (in-memory for simplicity, use Redis in production)
RATE_LIMIT = 100  # requests per minute
rate_limit_store = defaultdict(list)


# Middleware functions
async def request_logging_middleware(request: Request, call_next):
    """Middleware for request logging and request ID injection"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    logger.info(f"Request {request_id} started: {request.method} {request.url}")

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"Request {request_id} failed: {str(exc)}", exc_info=exc)
        raise
    finally:
        logger.info(f"Request {request_id} completed")

    return response


async def rate_limit_middleware(request: Request, call_next):
    """Basic rate limiting middleware"""
    # Skip metrics endpoint from rate limiting
    if request.url.path.startswith("/v1/core/metrics"):
        return await call_next(request)

    client_ip = request.client.host
    current_time = time.time()

    # Cleanup old requests
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if current_time - t < 60
    ]

    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded"},
        )

    rate_limit_store[client_ip].append(current_time)
    return await call_next(request)


# Create app first without middleware
app = FastAPI(
    title="Resume Matcher API",
    description="API for matching resumes to job postings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# Setup middleware stack
middlewares = [
    {
        "func": lambda app: app.middleware("http")(request_logging_middleware),
        "description": "Request logging and ID injection",
    },
    {
        "func": lambda app: app.middleware("http")(rate_limit_middleware),
        "description": "Rate limiting",
    },
    {
        "func": lambda app: app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        "description": "CORS support",
    },
]

# Apply middleware in order
for mw in middlewares:
    logger.info(f"Adding middleware: {mw['description']}")
    mw["func"](app)

# # Include routers (prefixes already include /v1)
# app.include_router(matching_router)
# app.include_router(learning_router)
# app.include_router(analytics_router)
# app.include_router(core_router)


# Exception handler remains the same
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    request_id = (
        request.state.request_id if hasattr(request.state, "request_id") else None
    )
    logger.error(f"Request {request_id} failed: {str(exc)}", exc_info=exc)

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(
                APIErrorResponse(
                    detail=exc.detail,
                    error_type=type(exc).__name__,
                    request_id=request_id,
                )
            ),
        )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=jsonable_encoder(
            APIErrorResponse(
                detail="Internal server error",
                error_type="InternalServerError",
                request_id=request_id,
            )
        ),
    )


# Clean up unused imports and functions
def clean_and_parse_json(json_str: str):
    """Helper to clean and parse JSON strings from form-data"""
    try:
        # Remove surrounding quotes if present
        if json_str.startswith('"') and json_str.endswith('"'):
            json_str = json_str[1:-1]
        # Unescape escaped characters
        json_str = json_str.replace('\\"', '"').replace("\\n", "\n")
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str[:200]}...")
        raise ValueError(f"Invalid JSON format: {str(e)}")


# Only now import and configure the instrumentator
from prometheus_fastapi_instrumentator import Instrumentator

# Configure instrumentation to only track specific paths
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/docs", "/openapi.json", "/health"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)

# Instrument the app but don't expose the endpoint (we're handling it manually)
instrumentator.instrument(app).expose(app, include_in_schema=False)
