import httpx
from fastapi import HTTPException, status
from jose import jwt, JWTError
from app.core.config import get_settings
from app.utils.logger import logger
from app.core.circuit_breaker import auth_service_breaker

settings = get_settings()


class AuthService:
    @staticmethod
    async def validate_token(token: str) -> dict:
        """
        Validate token by making a request to the auth service with circuit breaker
        """
        try:
            # Use circuit breaker for auth service calls
            return await auth_service_breaker.call(
                AuthService._make_auth_request, token
            )
        except Exception as e:
            # Fallback to local token validation if auth service is down
            logger.warning(f"Auth service unavailable, falling back to local validation: {e}")
            return AuthService.decode_token(token)
    
    @staticmethod
    async def _make_auth_request(token: str) -> dict:
        """Make the actual HTTP request to auth service"""
        try:
            # Create an async HTTP client with connection pooling
            timeout = httpx.Timeout(5.0, connect=2.0)
            async with httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            ) as client:
                # Make a request to the auth service's token validation endpoint
                response = await client.post(
                    f"{settings.AUTH_SERVICE_URL}{settings.API_V1_STR}/auth/validate-token",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    }
                )

                # Check response status
                if response.status_code == 200:
                    # Token is valid, return user data
                    return response.json()
                elif response.status_code == 401:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token",
                    )
                else:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail="Authentication service error",
                    )

        except httpx.RequestError as e:
            # Handle network-related errors
            logger.error(f"Auth service request error: {str(e)}")
            raise Exception(f"Authentication service unavailable: {str(e)}")
        except HTTPException:
            # Re-raise HTTP exceptions (401, etc.)
            raise
        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected auth error: {str(e)}")
            raise Exception(f"Unexpected authentication error: {str(e)}")

    @staticmethod
    def decode_token(token: str) -> dict:
        """
        Decode token locally as a fallback
        """
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
            )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
            )
