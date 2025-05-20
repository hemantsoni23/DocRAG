from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
import os

MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.chatbase

# Collection for chat logs
chat_logs_collection = db.chat_logs
feedback_collection = db.feedback
