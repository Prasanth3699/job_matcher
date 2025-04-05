from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from ..services.auth_service import AuthService

# Create an OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Dependency to get the current authenticated user
    """
    try:
        # Validate token via auth service
        user_data = await AuthService.validate_token(token)
        return user_data
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
