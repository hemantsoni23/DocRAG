from fastapi import APIRouter
from api.routes.auth import router as auth_router
from api.routes.chatbots import router as chatbots_router
from api.routes.documents import router as documents_router
from api.routes.chat import router as chat_router

router = APIRouter()

router.include_router(auth_router)
router.include_router(chatbots_router)
router.include_router(documents_router)
router.include_router(chat_router)