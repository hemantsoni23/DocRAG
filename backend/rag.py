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
from backend.vectorStore import get_vectorstore
from threading import Lock

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

    def _estimate_complexity(self, query: str) -> int:
        wc = len(query.split())
        return 3 if wc < 5 else 6 if wc < 15 else 10

    def get_documents(self, query: str) -> Tuple[List[Document], float]:
        adaptive_k = self._estimate_complexity(query)
        docs = self._try_retrieve(query, k=adaptive_k)

        if not docs:
            logger.warning("Retrying retrieval with larger k.")
            docs = self._try_retrieve(query, k=adaptive_k * 2)

        if not docs:
            logger.warning("Retrying retrieval with query optimization.")
            improved_query = self.optimizer.optimize(query)
            docs = self._try_retrieve(improved_query, k=adaptive_k * 2)

        if docs:
            compressed_context = self.compressor.compress(docs)
            if len(docs) <= 3:
                return docs, 0.7
            return self._rerank_with_llm(query, docs)

        return [], 0.0

    def _try_retrieve(self, query: str, k: int) -> List[Document]:
        try:
            return self.retriever.invoke(query, k=k)
        except Exception as e:
            logger.error(f"Retriever failed: {e}")
            return []

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
        return self.retriever.compressor.compress(docs), score

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
    def __init__(self, client_email=None):
        self.client_email = client_email
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

    def initialize_retriever(self, client_email=None):
        if client_email:
            self.client_email = client_email
        vectorstore = get_vectorstore(self.client_email)
        if not vectorstore:
            logger.warning(f"No vectorstore for {self.client_email}")
            return False
        self.retriever = AdaptiveRetriever(
            retriever=vectorstore.as_retriever(search_type="similarity"),
            llm=self.llm
        )
        logger.info(f"Retriever ready for {self.client_email}")
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

    def get_answer(self, query: str, chat_history: str | List[tuple[str, str]] = '', stream_callback=None, client_email: Optional[str] = None) -> Dict[str, Any]:
        if client_email and client_email != self.client_email:
            self.client_email = client_email
            self.initialize_retriever(client_email)
            self.setup_rag_chain()

        if not self.retriever or not self.qa_chain:
            if not self.initialize_retriever():
                return {"answer": "Please upload documents first.", "interaction_id": str(uuid.uuid4())}
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

            if relevance_score < 0.4:
                warning_msg = "⚠️ I'm not confident in the available information. The documents may be unrelated."
            elif relevance_score < 0.7:
                warning_msg = "Note: The retrieved context might only be partially relevant."

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
            logger.error(f"QA failure for client '{self.client_email}': {e}")
            answer = f"An error occurred: {str(e)}"

        self.interactions.append({
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "client_email": self.client_email,
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
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "helpful": helpful
        })
        logger.info(f"Feedback for {interaction_id} stored.")
        return True

    def get_system_stats(self, client_email=None):
        inters = [i for i in self.interactions if i.get("client_email") == client_email] if client_email else self.interactions
        feeds = [f for f in self.feedback_store if f.get("client_email") == client_email] if client_email else self.feedback_store

        if not feeds:
            return {"feedback_analysis": {"message": "No feedback yet."}}

        helpful_pct = (sum(1 for f in feeds if f["helpful"]) / len(feeds)) * 100
        avg_score = sum(f["score"] for f in feeds) / len(feeds)

        return {
            "feedback_analysis": {
                "total_interactions": len(inters),
                "feedback_count": len(feeds),
                "helpful_percentage": helpful_pct,
                "average_score": avg_score,
            }
        }

# Global RAG system cache
_rag_systems = {}
_rag_lock = Lock()

def get_rag_system(client_email=None):
    global _rag_systems
    client_email = client_email or "default"
    with _rag_lock:
        if client_email not in _rag_systems:
            _rag_systems[client_email] = SimpleRAGSystem(client_email)
    return _rag_systems[client_email]
