from fastapi import APIRouter, Form
from pydantic import EmailStr

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("", response_model=dict)
async def authenticate(email: EmailStr = Form(...)):
    """Authenticate a user with email."""
    return {"status": "success", "message": f"Logged in as {email}", "client_email": email}