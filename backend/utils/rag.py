#backend/rag.py
import os
import re
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from backend.utils.vectorStore import get_vectorstore
from threading import Lock
from functools import lru_cache

# Setup logging
logger = logging.getLogger(__name__)
load_dotenv()

class QueryOptimizer:
    def __init__(self, llm):
        self.llm = llm

    def optimize(self, query: str) -> str:
        prompt = f"""Improve or expand the following search query to retrieve more relevant documents, but keep it faithful:\nQuery: "{query}"\nImproved Query:"""
        try:
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return query

class ContextCompressor:
    def __init__(self, llm, max_tokens=3500):
        self.llm = llm
        self.max_tokens = max_tokens

    def compress(self, documents: List[Document]) -> str:
        combined = "\n\n".join(doc.page_content for doc in documents)
        if len(combined.split()) <= self.max_tokens:
            return combined
        logger.info(f"Context too long ({len(combined.split())} words). Compressing...")
        try:
            prompt = f"""Summarize the following documents into key points without losing important information:\n\n{combined}\n\nCompressed Summary:"""
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return combined

class AdaptiveRetriever:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.optimizer = QueryOptimizer(llm)
        self.compressor = ContextCompressor(llm)

    @lru_cache(maxsize=128)
    def get_documents(self, query: str) -> Tuple[List[Document], float]:
        docs = self._try_retrieve(query, k=4)

        if not docs:
            improved_query = self.optimizer.optimize(query)
            docs = self._try_retrieve(improved_query, k=6)

        if not docs:
            return [], 0.0

        score = self._estimate_relevance_score(query, self._combine_documents(docs))
        return docs, score

    def _try_retrieve(self, query: str, k: int) -> List[Document]:
        try:
            return self.retriever.invoke(query, k=k)
        except Exception as e:
            logger.error(f"Retriever failed: {e}")
            return []

    def _combine_documents(self, docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content.strip() for doc in docs)

    def _estimate_relevance_score(self, query: str, context: str) -> float:
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        common = query_words & context_words
        score = len(common) / max(len(query_words), 1)
        return min(max(score, 0.2), 1.0)

    def _rerank_with_llm(self, query: str, docs: List[Document]) -> Tuple[List[Document], float]:
        context = "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        prompt = f"""You are an expert AI evaluator.\nGiven the user query: "{query}"\nand these documents:\n\n{context}\n\nRank documents by relevance, output document numbers separated by commas. Also, rate overall document relevance to the query on a 0-1 scale (1=perfect).\nExample Output:\n"Ranking: 3,1,2\nRelevance Score: 0.85" """
        try:
            response = self.llm.invoke(prompt).strip()
            lines = response.splitlines()
            doc_line = next(line for line in lines if line.startswith("Ranking"))
            score_line = next(line for line in lines if line.startswith("Relevance Score"))

            indices = [int(n) - 1 for n in re.findall(r'\d+', doc_line)]
            score = float(score_line.split(":")[1].strip())
            return [docs[i] for i in indices if 0 <= i < len(docs)], score
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return docs, 0.5

class CustomRetrievalQAChain(LLMChain):
    retriever: AdaptiveRetriever

    def retrieve_context(self, query: str) -> Tuple[str, float]:
        docs, score = self.retriever.get_documents(query)
        compressed_context = self.retriever.compressor.compress(docs)
        return compressed_context, score

    def invoke(self, inputs: Dict[str, Any], run_manager=None):
        query = inputs["question"]
        chat_history = inputs.get("chat_history", "")
        context, _ = self.retrieve_context(query)
        return super().invoke({
            "context": context,
            "question": query,
            "chat_history": chat_history,
        }, run_manager=run_manager)

class SimpleRAGSystem:
    def __init__(self, client_email=None, chatbot_id="default"):
        self.client_email = client_email
        self.chatbot_id = chatbot_id
        self.llm = None
        self.retriever = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.feedback_store = []
        self.interactions = []
        self.initialize_llm()

    def initialize_llm(self):
        api_key = os.getenv("GEMINI_KEY")
        if not api_key:
            raise ValueError("GEMINI_KEY not found")
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

    def initialize_retriever(self, client_email=None, chatbot_id=None):
        if client_email:
            self.client_email = client_email
        if chatbot_id:
            self.chatbot_id = chatbot_id
            
        vectorstore = get_vectorstore(self.client_email, self.chatbot_id)
        if not vectorstore:
            logger.warning(f"No vectorstore for {self.client_email}, chatbot {self.chatbot_id}")
            return False
        self.retriever = AdaptiveRetriever(
            retriever=vectorstore.as_retriever(search_type="similarity"),
            llm=self.llm
        )
        logger.info(f"Retriever ready for {self.client_email}, chatbot {self.chatbot_id}")
        return True

    def setup_rag_chain(self):
        if not self.retriever:
            logger.warning("Retriever missing.")
            return False
        prompt = PromptTemplate(
            template="""
            You are a concise and accurate AI assistant. Use the context and chat history to answer the user's question.
            Chat History:
            {chat_history}

            Relevant Context:
            {context}

            Question:
            {question}

            Answer:
            """.strip(),
            input_variables=["context", "question", "chat_history"]
        )
        self.qa_chain = CustomRetrievalQAChain(
            llm=self.llm,
            prompt=prompt,
            retriever=self.retriever,
            verbose=False
        )
        logger.info("QA Chain ready.")
        return True

    def _format_chat_history(self, chat_history):
        if isinstance(chat_history, str):
            return chat_history
        return "\n\n".join(
            f"Human: {user}\nAssistant: {assistant or ''}" for user, assistant in chat_history
        ).strip()

    def _clean_response(self, response: str) -> str:
        blacklist = [
            "Based on the document", "According to the context", "From the documents",
            "Based on the context,", "Based on the information provided"
        ]
        for phrase in blacklist:
            response = response.replace(phrase, "")
        return response.strip()

    def get_answer(self, query: str, chat_history: str | List[tuple[str, str]] = '', stream_callback=None, client_email: Optional[str] = None, chatbot_id: Optional[str] = None) -> Dict[str, Any]:
        client_changed = client_email and client_email != self.client_email
        chatbot_changed = chatbot_id and chatbot_id != self.chatbot_id
        
        if client_changed or chatbot_changed:
            self.client_email = client_email or self.client_email
            self.chatbot_id = chatbot_id or self.chatbot_id
            self.initialize_retriever(self.client_email, self.chatbot_id)
            self.setup_rag_chain()

        if not self.retriever or not self.qa_chain:
            if not self.initialize_retriever():
                return {"answer": f"Please upload documents for chatbot '{self.chatbot_id}'.", "interaction_id": str(uuid.uuid4())}
            if not self.setup_rag_chain():
                return {"answer": "Could not initialize QA.", "interaction_id": str(uuid.uuid4())}

        interaction_id = str(uuid.uuid4())
        formatted_history = self._format_chat_history(chat_history)
        inputs = {"question": query, "chat_history": formatted_history}
        answer = ""
        relevance_score = 0.0
        warning_msg = ""

        try:
            context, relevance_score = self.qa_chain.retrieve_context(query)

            if relevance_score < 0.3:
                warning_msg = "⚠️ I'm not confident in the available information."
            else:
                warning_msg = ""

            if stream_callback:
                prompt_text = self.qa_chain.prompt.format(context=context, question=query, chat_history=formatted_history)
                for chunk in self.llm.stream(prompt_text):
                    if hasattr(chunk, 'text'):
                        token = chunk.text
                        answer += token
                        stream_callback(token)
            else:
                result = self.qa_chain.invoke(inputs)
                raw_answer = result.get("text", "")
                answer = self._clean_response(raw_answer)

            if warning_msg:
                answer = f"{warning_msg}\n\n{answer}"

        except Exception as e:
            logger.error(f"QA failure for client '{self.client_email}', chatbot '{self.chatbot_id}': {e}")
            answer = f"An error occurred: {str(e)}"

        self.interactions.append({
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "client_email": self.client_email,
            "chatbot_id": self.chatbot_id,
            "query": query,
            "answer": answer,
            "source_docs": [], 
            "relevance_score": relevance_score,
        })

        return {
            "answer": answer,
            "interaction_id": interaction_id
        }


    def add_feedback(self, interaction_id, score, helpful):
        interaction = next((i for i in self.interactions if i["interaction_id"] == interaction_id), None)
        if not interaction:
            logger.warning(f"No interaction {interaction_id} found.")
            return False
        self.feedback_store.append({
            "interaction_id": interaction_id,
            "client_email": interaction.get("client_email", self.client_email),
            "chatbot_id": interaction.get("chatbot_id", self.chatbot_id),
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "helpful": helpful
        })
        logger.info(f"Feedback for {interaction_id} stored.")
        return True

    def get_system_stats(self, client_email=None, chatbot_id=None):
        if client_email:
            if chatbot_id:
                # Stats for specific chatbot
                inters = [i for i in self.interactions if i.get("client_email") == client_email and i.get("chatbot_id") == chatbot_id]
                feeds = [f for f in self.feedback_store if f.get("client_email") == client_email and f.get("chatbot_id") == chatbot_id]
            else:
                # Stats for all client's chatbots
                inters = [i for i in self.interactions if i.get("client_email") == client_email]
                feeds = [f for f in self.feedback_store if f.get("client_email") == client_email]
        else:
            # All stats
            inters = self.interactions
            feeds = self.feedback_store

        if not feeds:
            chatbot_info = f" for chatbot '{chatbot_id}'" if chatbot_id else ""
            return {"feedback_analysis": {"message": f"No feedback yet{chatbot_info}."}}

        helpful_pct = (sum(1 for f in feeds if f["helpful"]) / len(feeds)) * 100
        avg_score = sum(f["score"] for f in feeds) / len(feeds)

        return {
            "feedback_analysis": {
                "total_interactions": len(inters),
                "interactions_with_feedback": len(feeds),
                "helpful_percentage": helpful_pct,
                "average_score": avg_score,
            }
        }

# Global RAG system cache
# Structure: {client_email: {chatbot_id: rag_system}}
_rag_systems = {}
_rag_lock = Lock()

def get_rag_system(client_email=None, chatbot_id="default"):
    global _rag_systems
    client_email = client_email or "default"
    
    with _rag_lock:
        if client_email not in _rag_systems:
            _rag_systems[client_email] = {}
        
        if chatbot_id not in _rag_systems[client_email]:
            _rag_systems[client_email][chatbot_id] = SimpleRAGSystem(client_email, chatbot_id)
    
    return _rag_systems[client_email][chatbot_id]