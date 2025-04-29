# api/routes/feedback.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from backend.rag import get_rag_system

router = APIRouter()
logger = logging.getLogger(__name__)

class FeedbackRequest(BaseModel):
    score: int
    helpful: bool
    chat_state: dict = {}
    client_email: str

class FeedbackResponse(BaseModel):
    message: str

@router.post("/", response_model=FeedbackResponse)
def submit_feedback(feedback: FeedbackRequest):
    if not feedback.chat_state.get("current_interaction_id"):
        raise HTTPException(status_code=400, detail="No interaction to provide feedback for.")
    if not feedback.client_email:
        raise HTTPException(status_code=400, detail="Email not set. Please log in first.")
    
    try:
        rag_system = get_rag_system(feedback.client_email)
        success = rag_system.add_feedback(feedback.chat_state.get("current_interaction_id"), feedback.score, feedback.helpful)
        if success:
            return FeedbackResponse(message="✅ Feedback submitted! Thank you.")
        else:
            return FeedbackResponse(message="❌ Failed to record feedback.")
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

class StatsRequest(BaseModel):
    client_email: str

class StatsResponse(BaseModel):
    statistics: str

@router.post("/stats", response_model=StatsResponse)
def get_system_stats(stats_req: StatsRequest):
    if not stats_req.client_email:
        raise HTTPException(status_code=400, detail="Email not set. Please log in first.")
    try:
        rag_system = get_rag_system(stats_req.client_email)
        stats = rag_system.get_system_stats(stats_req.client_email)
        feedback = stats.get("feedback_analysis", {})
        if "message" in feedback:
            return StatsResponse(statistics=feedback["message"])
        results = f"""
## 📊 System Statistics for {stats_req.client_email}

**Interactions:**
- Total: {feedback.get('total_interactions', 0)}
- With feedback: {feedback.get('interactions_with_feedback', 0)}

**Quality:**
- Helpful responses: {feedback.get('helpful_percentage', 0):.1f}%
- Average score: {feedback.get('average_score', 0):.1f}/5

**Learning:**
- Documents with feedback: {stats.get('documents_with_feedback', 0)}
"""
        return StatsResponse(statistics=results)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")
