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
from app.middleware.subscription_middleware import verify_subscription

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


app.middleware("http")


async def subscription_middleware(request: Request, call_next):
    # Only apply to matching endpoints
    if request.url.path.startswith("/api/v1/matching/new-matchs"):
        # Get the authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required"},
            )

        # Extract token
        token = auth_header.replace("Bearer ", "")

        try:
            # Use the verify_subscription logic
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer", credentials=token
            )
            await verify_subscription(request, credentials)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    # Continue with the request
    response = await call_next(request)
    return response
