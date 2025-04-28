import json
import logging
from backend.rag import get_rag_system

logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, testset_path: str):
        """
        Initialize evaluator with a JSON test set.
        Each item should be {"query": "question text", "ideal_answer": "expected text"}
        """
        self.rag_system = get_rag_system()
        self.testset_path = testset_path
        self.load_testset()
    
    def load_testset(self):
        with open(self.testset_path, "r") as f:
            self.testset = json.load(f)
        logger.info(f"Loaded {len(self.testset)} test queries for evaluation.")

    def simple_similarity_score(self, generated_answer: str, ideal_answer: str) -> float:
        """A simple word-overlap based similarity score (normalized)"""
        gen_words = set(generated_answer.lower().split())
        ideal_words = set(ideal_answer.lower().split())
        if not ideal_words:
            return 0.0
        overlap = len(gen_words.intersection(ideal_words))
        return overlap / len(ideal_words)

    def evaluate(self, similarity_threshold=0.6):
        """
        Evaluate RAG on the test set.
        
        Args:
            similarity_threshold: float (minimum similarity to count as correct)
        
        Returns:
            Dictionary with evaluation results.
        """
        total = len(self.testset)
        correct = 0
        scores = []

        for item in self.testset:
            query = item["query"]
            ideal_answer = item["ideal_answer"]

            # Get answer from RAG system
            response = self.rag_system.get_answer(query)
            generated_answer = response["answer"]
            print(generated_answer) 

            score = self.simple_similarity_score(generated_answer, ideal_answer)
            scores.append(score)

            if score >= similarity_threshold:
                correct += 1

            logger.info(f"Query: {query}")
            logger.info(f"Ideal: {ideal_answer}")
            logger.info(f"Generated: {generated_answer}")
            logger.info(f"Similarity Score: {score:.2f}")
            logger.info("-" * 60)

        accuracy = correct / total * 100
        avg_score = sum(scores) / max(total, 1)

        result = {
            "total_queries": total,
            "correct_predictions": correct,
            "accuracy_percentage": round(accuracy, 2),
            "average_similarity_score": round(avg_score, 3),
        }
        return result

if __name__ == "__main__":
    # Example usage
    evaluator = RAGEvaluator(testset_path="./backend/testset.json")
    results = evaluator.evaluate()
    print("Evaluation Results:", results)
