# middleware/subscription_middleware.py
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import requests
from datetime import datetime
from ..utils.logger import logger

from ..core.config import settings

# Set up logging


# Security scheme
security = HTTPBearer()


async def verify_subscription(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Middleware to verify user has an active subscription
    to access protected ML endpoints
    """
    try:
        # Extract token
        token = credentials.credentials

        # First, validate the token with the Auth Service
        # This ensures the token is valid and identifies the user
        auth_response = requests.post(
            f"{settings.AUTH_SERVICE_URL}/api/v1/auth/validate-token",
            headers={"Authorization": f"Bearer {token}"},
        )

        if auth_response.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
            )

        # Get user details from auth response
        user_data = auth_response.json()
        user_id = user_data.get("user_id")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid user data in token",
            )

        # Now check subscription status with Subscription Service
        subscription_response = requests.get(
            f"{settings.SUBSCRIPTION_SERVICE_URL}/api/v1/subscriptions/status",
            headers={"Authorization": f"Bearer {token}"},
        )

        if subscription_response.status_code != 200:
            logger.error(
                f"Subscription validation failed: {subscription_response.text}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Subscription validation failed",
            )

        subscription_data = subscription_response.json()

        # Check if user has an active subscription
        if not subscription_data.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Active subscription required to access this endpoint",
            )

        # Add subscription data to request state for use in endpoint handlers
        request.state.subscription = subscription_data.get("subscription")
        request.state.user = user_data

        return user_data

    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token format"
        )
    except requests.RequestException as e:
        logger.error(f"Service communication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service communication error",
        )
