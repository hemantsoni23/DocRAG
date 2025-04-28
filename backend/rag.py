# backend/rag.py

import os
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from backend.vectorStore import get_vectorstore
from threading import Lock

# Setup logging
logger = logging.getLogger(__name__)
load_dotenv()

class CustomRetrievalQAChain(LLMChain):
    """Chain that handles retrieval, memory-aware prompting, and response."""
    retriever: Any

    def retrieve_context(self, query: str) -> str:
        docs = self.retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, inputs: Dict[str, Any], run_manager=None):
        query = inputs["question"]
        chat_history = inputs.get("chat_history", "")
        context = self.retrieve_context(query)

        prompt_inputs = {
            "context": context,
            "question": query,
            "chat_history": chat_history,
        }
        return super().invoke(prompt_inputs, run_manager=run_manager)

class SimpleRAGSystem:
    def __init__(self):
        self.llm = None
        self.retriever = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.feedback_store = []
        self.interactions = []
        
        self.initialize_llm()
        self.initialize_retriever()
        self.setup_rag_chain()

    def initialize_llm(self):
        api_key = os.getenv("GEMINI_KEY")
        if not api_key:
            raise ValueError("GEMINI_KEY not found in environment variables")

        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=api_key,
            temperature=0.2,
            top_p=0.85,
            top_k=40,
            max_output_tokens=2048,
            streaming=True,
        )
        logger.info("LLM initialized.")

    def initialize_retriever(self):
        vectorstore = get_vectorstore()
        if not vectorstore:
            raise ValueError("Vector store is not initialized.")
        
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        logger.info("Retriever initialized.")

    def setup_rag_chain(self):
        template = """
You are a helpful AI assistant answering questions based on the given documents and previous conversation.
If the answer is not found in the provided context, politely state that you don't have enough information.

CHAT HISTORY:
{chat_history}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ASSISTANT RESPONSE:
"""
        prompt = PromptTemplate(
            template=template.strip(),
            input_variables=["context", "question", "chat_history"]
        )
        self.qa_chain = CustomRetrievalQAChain(
            llm=self.llm,
            prompt=prompt,
            retriever=self.retriever,
            verbose=False
        )
        logger.info("RAG QA chain setup complete.")

    def _format_chat_history(self, chat_history):
        if isinstance(chat_history, str):
            return chat_history
        if not chat_history:
            return ""
        
        return "\n\n".join(
            f"Human: {user}\nAssistant: {assistant or ''}"
            for user, assistant in chat_history
        ).strip()

    def get_answer(self, query, chat_history='', stream_callback=None):
        if not self.qa_chain:
            self.setup_rag_chain()

        logger.info(f"Processing query: {query}")
        interaction_id = str(uuid.uuid4())
        formatted_history = self._format_chat_history(chat_history)

        inputs = {
            "question": query,
            "chat_history": formatted_history,
        }

        answer = ""
        try:
            if stream_callback:
                context = self.qa_chain.retrieve_context(query)
                prompt_text = self.qa_chain.prompt.format(
                    context=context,
                    question=query,
                    chat_history=formatted_history
                )
                for chunk in self.llm.stream(prompt_text):
                    if hasattr(chunk, 'text'):
                        token = chunk.text
                        answer += token
                        stream_callback(token)
            else:
                result = self.qa_chain.invoke(inputs)
                answer = result.get("text", "")
        except Exception as e:
            logger.error(f"Error during answer retrieval: {e}")
            answer = f"An error occurred: {str(e)}"

        self.interactions.append({
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "source_docs": [],
        })

        return {
            "answer": answer,
            "interaction_id": interaction_id
        }

    def add_feedback(self, interaction_id, score, helpful):
        interaction = next((i for i in self.interactions if i["interaction_id"] == interaction_id), None)
        if not interaction:
            logger.warning(f"Interaction {interaction_id} not found for feedback.")
            return False
        
        self.feedback_store.append({
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "helpful": helpful
        })
        logger.info(f"Feedback recorded for interaction {interaction_id}")
        return True

    def get_system_stats(self):
        total_interactions = len(self.interactions)
        feedback_entries = len(self.feedback_store)

        if feedback_entries == 0:
            return {"feedback_analysis": {"message": "No feedback yet."}}

        helpful_count = sum(1 for f in self.feedback_store if f["helpful"])
        avg_score = sum(f["score"] for f in self.feedback_store) / feedback_entries
        helpful_percentage = (helpful_count / feedback_entries) * 100

        return {
            "feedback_analysis": {
                "total_interactions": total_interactions,
                "interactions_with_feedback": feedback_entries,
                "helpful_percentage": helpful_percentage,
                "average_score": avg_score,
            }
        }

# Singleton RAG system (thread-safe lazy init)
_rag_system = None
_rag_lock = Lock()

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        with _rag_lock:
            if _rag_system is None:  # Double-checked locking
                _rag_system = SimpleRAGSystem()
    return _rag_system
