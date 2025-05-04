import httpx
from fastapi import HTTPException, status
from jose import jwt, JWTError
from app.core.config import get_settings
from app.utils.logger import logger

settings = get_settings()


class AuthService:
    @staticmethod
    async def validate_token(token: str) -> dict:
        """
        Validate token by making a request to the auth service
        """
        try:
            # Create an async HTTP client
            async with httpx.AsyncClient() as client:
                # Make a request to the auth service's token validation endpoint
                response = await client.post(
                    f"{settings.AUTH_SERVICE_URL}{settings.API_V1_STR}/auth/validate-token",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    timeout=5.0,  # 5 seconds timeout
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
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Authentication service unavailable: {str(e)}",
            )
        except Exception as e:
            # Catch any unexpected errors
            logger.error(f"Unexpected auth error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected authentication error: {str(e)}",
            )

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
