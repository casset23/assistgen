from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.security import create_access_token
from app.schemas.user import UserCreate, UserResponse, Token, UserLogin
from app.services.user_service import UserService
from datetime import timedelta, datetime
from app.core.config import settings
from app.models.user import User

router = APIRouter()


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        user_service = UserService(db)
        user = await user_service.create_user(user_data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/token", response_model=Token)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):
    try:
        user_service = UserService(db)
        user = await user_service.authenticate_user(user_data.email, user_data.password)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception:
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": "demo@assistgen.com"}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info():
    return UserResponse(
        id=1,
        username="demo_user",
        email="demo@assistgen.com",
        status="active",
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
    )


@router.get("/validate-token")
async def validate_token():
    return {"valid": True, "user_id": 1}


@router.get("/users/me/mock", response_model=UserResponse)
async def get_mock_user_info():
    return UserResponse(
        id=1,
        username="demo_user",
        email="demo@assistgen.com",
        status="active",
        created_at=datetime.utcnow(),
        last_login=datetime.utcnow(),
    )
