import os
import json
import logging
import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    timestamp: str
    query: str
    response: str
    retrieved_docs: List[str]
    feedback_score: int = 0
    is_helpful: bool = False

class FeedbackSystem:
    """RL Feedback System for RAG optimization."""

    def __init__(self, feedback_dir="./backend/feedback"):
        self.feedback_dir = feedback_dir
        self.feedback_file = os.path.join(feedback_dir, "feedback_data.json")
        self.interactions: List[Interaction] = []
        self.load_feedback_data()

    def load_feedback_data(self):
        """Load existing feedback."""
        try:
            os.makedirs(self.feedback_dir, exist_ok=True)
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, "r") as f:
                    data = json.load(f)
                self.interactions = [Interaction(**item) for item in data]
                logger.info(f"✅ Loaded {len(self.interactions)} feedback entries.")
        except Exception as e:
            logger.error(f"❌ Failed loading feedback: {e}")

    def save_feedback_data(self):
        """Persist feedback to disk."""
        try:
            with open(self.feedback_file, "w") as f:
                json.dump([asdict(i) for i in self.interactions], f, indent=2)
            logger.info(f"✅ Saved {len(self.interactions)} feedback entries.")
        except Exception as e:
            logger.error(f"❌ Failed saving feedback: {e}")

    def log_interaction(self, query: str, response: str, retrieved_docs: List[str]) -> str:
        """Log a new interaction."""
        timestamp = datetime.datetime.now().isoformat()
        interaction = Interaction(timestamp, query, response, retrieved_docs)
        self.interactions.append(interaction)
        self.save_feedback_data()
        return timestamp

    def add_feedback(self, timestamp: str, feedback_score: int, is_helpful: bool) -> bool:
        """Attach feedback to an interaction."""
        for interaction in self.interactions:
            if interaction.timestamp == timestamp:
                interaction.feedback_score = feedback_score
                interaction.is_helpful = is_helpful
                self.save_feedback_data()
                return True
        logger.warning(f"⚠️ Interaction {timestamp} not found.")
        return False

    def get_similar_queries(self, query: str, threshold: int = 3) -> List[Dict[str, Any]]:
        """Simple similarity match based on word overlap."""
        query_words = set(query.lower().split())
        results = [
            asdict(i) for i in self.interactions
            if (i.feedback_score >= 4 or i.is_helpful)
            and len(query_words.intersection(set(i.query.lower().split()))) >= threshold
        ]
        return results

    def analyze_feedback(self) -> Dict[str, Any]:
        """Analyze feedback for reporting."""
        if not self.interactions:
            return {"message": "No feedback data available."}
        
        total = len(self.interactions)
        with_feedback = sum(1 for i in self.interactions if i.feedback_score > 0)
        helpful = sum(1 for i in self.interactions if i.is_helpful)
        avg_score = sum(i.feedback_score for i in self.interactions if i.feedback_score > 0) / max(1, with_feedback)
        
        return {
            "total_interactions": total,
            "interactions_with_feedback": with_feedback,
            "helpful_responses": helpful,
            "helpful_percentage": helpful / max(1, with_feedback) * 100,
            "average_score": round(avg_score, 2)
        }
