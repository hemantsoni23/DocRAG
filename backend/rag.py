import os
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from backend.vectorStore import get_vectorstore
from threading import Lock

# Setup logging
logger = logging.getLogger(__name__)
load_dotenv()

class QueryOptimizer:
    """Optional: Use LLM to rephrase or expand query if retrieval fails."""
    def __init__(self, llm):
        self.llm = llm

    def optimize(self, query: str) -> str:
        prompt = f"""
        Improve or expand the following search query to retrieve more relevant documents, but keep it faithful:
        Query: "{query}"
        Improved Query:
        """
        try:
            optimized = self.llm.invoke(prompt).strip()
            return optimized
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return query

class ContextCompressor:
    """Compress context if needed to stay within token limits."""
    def __init__(self, llm, max_tokens=3500):
        self.llm = llm
        self.max_tokens = max_tokens

    def compress(self, documents: List[Document]) -> str:
        combined = "\n\n".join(doc.page_content for doc in documents)
        if len(combined.split()) <= self.max_tokens:
            return combined
        logger.info(f"Context too long ({len(combined.split())} words). Compressing...")
        try:
            compression_prompt = f"""
            Summarize the following documents into key points without losing important information:

            {combined}

            Compressed Summary:
            """
            summary = self.llm.invoke(compression_prompt).strip()
            return summary
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return combined

class AdaptiveRetriever:
    """Advanced Adaptive Retriever with dynamic k, reranking, fallback, optimization."""
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.optimizer = QueryOptimizer(llm)
        self.compressor = ContextCompressor(llm)

    def _estimate_complexity(self, query: str) -> int:
        wc = len(query.split())
        if wc < 5:
            return 3
        elif wc < 15:
            return 6
        else:
            return 10

    def get_documents(self, query: str) -> List[Document]:
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
            reranked_docs, _ = self._rerank_with_llm(query, docs)
            return reranked_docs
        return []

    def _try_retrieve(self, query: str, k: int) -> List[Document]:
        try:
            return self.retriever.invoke(query, k=k)
        except Exception as e:
            logger.error(f"Retriever failed: {e}")
            return []

    def _rerank_with_llm(self, query: str, docs: List[Document]) -> (List[Document], float):
        context = "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        rerank_prompt = f"""
        You are an expert AI evaluator.
        Given the user query: "{query}"
        and these documents:

        {context}

        Rank documents by relevance, output document numbers separated by commas. Also, rate overall document relevance to the query on a 0-1 scale (1=perfect).
        Example Output:
        "Ranking: 3,1,2
        Relevance Score: 0.85"
        """

        try:
            response = self.llm.invoke(rerank_prompt).strip()
            lines = response.splitlines()
            doc_line = next(line for line in lines if line.startswith("Ranking"))
            score_line = next(line for line in lines if line.startswith("Relevance Score"))

            doc_indices = [int(num.strip()) - 1 for num in doc_line.split(":")[1].split(",")]
            relevance_score = float(score_line.split(":")[1].strip())

            reranked_docs = [docs[i] for i in doc_indices if 0 <= i < len(docs)]
            return reranked_docs, relevance_score
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return docs, 0.5

class CustomRetrievalQAChain(LLMChain):
    retriever: AdaptiveRetriever

    def retrieve_context(self, query: str) -> str:
        docs = self.retriever.get_documents(query)
        compressed_context = self.retriever.compressor.compress(docs)
        return compressed_context

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
    def __init__(self, client_email=None):
        self.client_email = client_email
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
        template = """
        You are a helpful AI assistant using retrieved documents and prior conversation.
        Answer strictly based on context. If unsure, say "I don't know."

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
        logger.info("QA Chain ready.")
        return True

    def _format_chat_history(self, chat_history):
        if isinstance(chat_history, str):
            return chat_history
        return "\n\n".join(
            f"Human: {user}\nAssistant: {assistant or ''}"
            for user, assistant in chat_history
        ).strip()

    def get_answer(self, query, chat_history='', stream_callback=None, client_email=None):
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
            logger.error(f"QA failure: {e}")
            answer = f"An error occurred: {str(e)}"

        self.interactions.append({
            "interaction_id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "client_email": self.client_email,
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
        if client_email:
            inters = [i for i in self.interactions if i.get("client_email") == client_email]
            feeds = [f for f in self.feedback_store if f.get("client_email") == client_email]
        else:
            inters = self.interactions
            feeds = self.feedback_store

        if not feeds:
            return {"feedback_analysis": {"message": "No feedback yet."}}

        helpful_count = sum(1 for f in feeds if f["helpful"])
        avg_score = sum(f["score"] for f in feeds) / len(feeds)
        helpful_pct = (helpful_count / len(feeds)) * 100

        return {
            "feedback_analysis": {
                "total_interactions": len(inters),
                "feedback_count": len(feeds),
                "helpful_percentage": helpful_pct,
                "average_score": avg_score,
            }
        }

_rag_systems = {}
_rag_lock = Lock()

def get_rag_system(client_email=None):
    global _rag_systems
    if client_email is None:
        client_email = "default"
    if client_email not in _rag_systems:
        with _rag_lock:
            if client_email not in _rag_systems:
                _rag_systems[client_email] = SimpleRAGSystem(client_email)
    return _rag_systems[client_email]