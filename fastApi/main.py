from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastApi.routes import documents, chat, feedback

app = FastAPI(title="RAG Backend API", description="API endpoints for document processing, chat and feedback.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
