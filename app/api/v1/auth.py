from fastapi import APIRouter, HTTPException
from app.models.user import User
from app.schemas.auth import UserRegister, UserLogin, TokenOut, UserOut
from app.core.security import hash_password, verify_password, create_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut, status_code=201)
async def register(body: UserRegister):
    # Check email is not already taken
    existing = await User.find_one(User.email == body.email)
    if existing:
        raise HTTPException(400, "Email already registered")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
        timezone=body.timezone,
    )
    await user.insert()

    return TokenOut(access_token=create_token(str(user.id)))


@router.post("/login", response_model=TokenOut)
async def login(body: UserLogin):
    user = await User.find_one(User.email == body.email)

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    return TokenOut(access_token=create_token(str(user.id)))


@router.get("/me", response_model=UserOut)
async def me(current_user: User = Depends(get_current_user)):
    return UserOut(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
        timezone=current_user.timezone,
        tier=current_user.tier,
    )
