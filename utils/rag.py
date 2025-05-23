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
from utils.vectorStore import get_vectorstore
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
    def __init__(self, retriever, llm_main, llm_fast=None):
        self.retriever = retriever
        self.llm_main = llm_main
        self.llm_fast = llm_fast or llm_main
        self.optimizer = QueryOptimizer(self.llm_fast)
        self.compressor = ContextCompressor(self.llm_main)

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
        # Enhanced relevance scoring
        query_words = set(query.lower().split())
        context_words = set(context.lower().split())
        common = query_words & context_words
        
        # Base score from word overlap
        base_score = len(common) / max(len(query_words), 1)
        
        # Boost score if query contains specific terms found in context
        query_lower = query.lower()
        context_lower = context.lower()
        
        # Check for exact phrase matches
        query_phrases = [phrase.strip() for phrase in query_lower.split() if len(phrase.strip()) > 3]
        phrase_matches = sum(1 for phrase in query_phrases if phrase in context_lower)
        phrase_boost = (phrase_matches / max(len(query_phrases), 1)) * 0.3
        
        final_score = min(base_score + phrase_boost, 1.0)
        return max(final_score, 0.1)  # Minimum score to avoid complete zeros

    def _rerank_with_llm(self, query: str, docs: List[Document]) -> Tuple[List[Document], float]:
        context = "\n\n".join([f"Doc {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
        prompt = f"""You are an expert AI evaluator.\nGiven the user query: "{query}"\nand these documents:\n\n{context}\n\nRank documents by relevance, output document numbers separated by commas. Also, rate overall document relevance to the query on a 0-1 scale (1=perfect).\nExample Output:\n"Ranking: 3,1,2\nRelevance Score: 0.85" """
        try:
            response = self.llm_fast.invoke(prompt).strip()
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
        context, relevance_score = self.retrieve_context(query)
        
        # Pass relevance score for decision making
        return super().invoke({
            "context": context,
            "question": query,
            "chat_history": chat_history,
            "relevance_score": relevance_score,
        }, run_manager=run_manager)

class SimpleRAGSystem:
    def __init__(self, client_email=None, chatbot_id="default"):
        self.client_email = client_email
        self.chatbot_id = chatbot_id
        self.llm_main = None
        self.llm_fast = None
        self.retriever = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.feedback_store = []
        self.interactions = []
        # Set minimum relevance threshold for answering questions
        self.min_relevance_threshold = 0.4  # Increased from 0.3
        self.initialize_models()

    def initialize_models(self):
        main_api_key = os.getenv("GEMINI_KEY")
        if not main_api_key:
            raise ValueError("GEMINI_KEY not found")
        
        self.llm_main = GoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            google_api_key=main_api_key,
            temperature=0.1,  # Reduced temperature for more focused responses
            top_p=0.8,        # Reduced for more focused responses
            top_k=20,         # Reduced for more focused responses
            max_output_tokens=2048,
        )
        
        fast_api_key = os.getenv("FAST_MODEL_KEY", main_api_key)
        
        self.llm_fast = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=fast_api_key,
            temperature=0.05,  # Very low temperature for deterministic tasks
            top_p=0.9,
            top_k=40,
            max_output_tokens=1024,  
        )
        
        logger.info("LLM models initialized.")

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
            llm_main=self.llm_main,
            llm_fast=self.llm_fast
        )
        logger.info(f"Retriever ready for {self.client_email}, chatbot {self.chatbot_id}")
        return True

    def setup_rag_chain(self):
        if not self.retriever:
            logger.warning("Retriever missing.")
            return False
            
        # Updated prompt with more conversational and friendly tone
        prompt = PromptTemplate(
            template="""
            You are a helpful, friendly, and conversational AI assistant created by Mapaman. You are designed to help users explore and understand the documents in your knowledge base.

            IMPORTANT GUIDELINES:
            - Be warm, friendly, and conversational in your responses
            - Use the provided context to answer questions about the documents
            - If you can answer from the context, provide a complete and helpful response
            - If the context doesn't contain enough information, politely explain that you don't have that information in your document collection
            - Always maintain a helpful and engaging tone
            - Encourage users to ask more questions about the documents

            Chat History:
            {chat_history}

            Relevant information from your document collection:
            {context}

            User's question: {question}

            Instructions: Answer the user's question in a friendly, conversational way. If you can find relevant information in the context above, use it to provide a helpful response. If the context doesn't contain enough information to answer the question, politely let them know and encourage them to ask about topics covered in your documents.

            Response:
            """.strip(),
            input_variables=["context", "question", "chat_history"]
        )
        
        self.qa_chain = CustomRetrievalQAChain(
            llm=self.llm_main,
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

    def _classify_query_type(self, query: str) -> str:
        """
        Classify query into different types for appropriate handling
        Returns: 'greeting', 'capability', 'conversational', 'factual_out_of_scope', 'factual_in_scope'
        """
        query_lower = query.lower().strip()
        
        # Greeting patterns
        greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening|greetings)',
            r'^(what\'s up|how are you|how\'s it going)',
            r'(hi there|hey there|hello there)',
        ]
        
        # Capability/meta questions
        capability_patterns = [
            r'what can you do',
            r'how can you help',
            r'what are you capable of',
            r'what is your purpose',
            r'what are your abilities',
            r'how do you work',
            r'what kind of questions can i ask',
            r'what do you know about',
        ]
        
        # General conversational (but not when followed by actual questions)
        conversational_patterns = [
            r'^(thanks|thank you|appreciated)(?!\s+.+)',  # Thanks but not "thanks for telling me about X"
            r'^(ok|okay|alright|sure|fine)(?!\s+(?:yes|tell|show|explain|what|who|how|where|when|why))',  # Ok but not "ok yes tell me" or "ok what about"
            r'^(good|great|nice|cool|awesome)(?!\s+.+)',  # Good but not "good question about X"
            r'(how was your day|tell me about yourself)',
            r'^(bye|goodbye|see you|farewell)(?!\s+.+)',  # Goodbye but not "goodbye, but first tell me"
        ]
        
        # Factual questions that are clearly out of scope (general knowledge)
        factual_out_of_scope_patterns = [
            r"who\s+(?:is|are|was|were)\s+(?:monkey\s+d\.?\s+luffy|naruto|goku|superman|batman|spiderman)",
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:weather|time|date|news)",
            r"tell\s+me\s+about\s+(?:python|javascript|programming|cooking|sports)",
            r"how\s+to\s+(?:cook|program|drive|fly)",
            r"what\s+(?:is|are)\s+(?:the\s+)?capital\s+of",
            r"when\s+(?:was|were|did)",
            r"where\s+(?:is|are|was|were)",
        ]
        
    def _classify_query_type(self, query: str) -> str:
        """
        Classify query into different types for appropriate handling
        Returns: 'greeting', 'capability', 'conversational', 'factual_out_of_scope', 'factual_in_scope'
        """
        query_lower = query.lower().strip()
        
        # First check if it's a continuation/follow-up question
        continuation_patterns = [
            r'(?:ok|okay|alright|sure|yes)\s+(?:yes|tell|show|explain|what|who|how|where|when|why)',
            r'(?:ok|okay|alright|sure|yes)\s+.+(?:about|more|tell|explain|show)',
            r'(?:and|also|additionally)\s+(?:what|who|how|where|when|why)',
            r'tell me more',
            r'explain (?:more|further|that)',
            r'what (?:about|else)',
            r'anything else about',
        ]
        
        for pattern in continuation_patterns:
            if re.search(pattern, query_lower):
                return 'factual_in_scope'  # Treat as factual question
        
        # Greeting patterns
        greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon|good evening|greetings)',
            r'^(what\'s up|how are you|how\'s it going)',
            r'(hi there|hey there|hello there)',
        ]
        
        # Capability/meta questions
        capability_patterns = [
            r'what can you do',
            r'how can you help',
            r'what are you capable of',
            r'what is your purpose',
            r'what are your abilities',
            r'how do you work',
            r'what kind of questions can i ask',
            r'what do you know about',
        ]
        
        # General conversational (but not when followed by actual questions)
        conversational_patterns = [
            r'^(thanks|thank you|appreciated)(?!\s+.+)',  # Thanks but not "thanks for telling me about X"
            r'^(ok|okay|alright|sure|fine)(?!\s+(?:yes|tell|show|explain|what|who|how|where|when|why))',  # Ok but not "ok yes tell me" or "ok what about"
            r'^(good|great|nice|cool|awesome)(?!\s+.+)',  # Good but not "good question about X"
            r'(how was your day|tell me about yourself)',
            r'^(bye|goodbye|see you|farewell)(?!\s+.+)',  # Goodbye but not "goodbye, but first tell me"
        ]
        
        # Factual questions that are clearly out of scope (general knowledge)
        factual_out_of_scope_patterns = [
            r"who\s+(?:is|are|was|were)\s+(?:monkey\s+d\.?\s+luffy|naruto|goku|superman|batman|spiderman)",
            r"what\s+(?:is|are)\s+(?:the\s+)?(?:weather|time|date|news)",
            r"tell\s+me\s+about\s+(?:python|javascript|programming|cooking|sports)",
            r"how\s+to\s+(?:cook|program|drive|fly)",
            r"what\s+(?:is|are)\s+(?:the\s+)?capital\s+of",
            r"when\s+(?:was|were|did)",
            r"where\s+(?:is|are|was|were)",
        ]
        
        # Check patterns in order of priority
        for pattern in greeting_patterns:
            if re.search(pattern, query_lower):
                return 'greeting'
                
        for pattern in capability_patterns:
            if re.search(pattern, query_lower):
                return 'capability'
                
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                return 'conversational'
                
        for pattern in factual_out_of_scope_patterns:
            if re.search(pattern, query_lower):
                return 'factual_out_of_scope'
        
        # Check for creator questions separately
        creator_patterns = [
            r"who\s+(?:is|are|was|were)\s+(?:the\s+)?creator",
            r"who\s+(?:made|created|built|developed)\s+you",
            r"what\s+(?:is|are)\s+your\s+(?:name|creator|maker)",
        ]
        
        for pattern in creator_patterns:
            if re.search(pattern, query_lower):
                return 'factual_out_of_scope'  # Handle as out of scope but with creator response
        
        # Default to factual (will be checked against context)
        return 'factual_in_scope'

    def _is_out_of_scope(self, query: str, context: str, relevance_score: float, query_type: str) -> bool:
        """
        Determine if a query is out of scope based on query type and relevance score
        """
        # Never consider conversational queries as out of scope
        if query_type in ['greeting', 'capability', 'conversational']:
            return False
            
        # Factual questions that are clearly out of scope
        if query_type == 'factual_out_of_scope':
            return True
            
        # For factual questions, check relevance score
        if query_type == 'factual_in_scope' and relevance_score < self.min_relevance_threshold:
            return True
                    
        return False

    def _generate_conversational_response(self, query: str, query_type: str) -> str:
        """Generate appropriate conversational responses"""
        
        if query_type == 'greeting':
            greetings = [
                "Hello! Great to meet you! I'm here to help answer questions about the documents in my knowledge base. What would you like to know?",
                "Hi there! Nice to chat with you! I'm ready to help you explore the information from the uploaded documents. What can I assist you with?",
                "Hey! Thanks for stopping by! I'm here to help you find answers from the documents that have been shared with me. What are you curious about?",
                "Hello! I'm excited to help you today! I can answer questions based on the documents in my knowledge base. What would you like to discover?",
                "Hi! It's wonderful to connect with you! I'm here to help you navigate through the information in my document collection. How can I assist you?",
            ]
            return greetings[hash(query) % len(greetings)]
            
        elif query_type == 'capability':
            capabilities = [
                "I'm here to help you explore and understand the documents that have been uploaded to my knowledge base! I can answer questions, explain concepts, find specific information, and discuss topics covered in those documents. What would you like to know about?",
                "Great question! I'm designed to be your friendly guide through the document collection in my knowledge base. I can help you find answers, clarify information, summarize content, and discuss any topics covered in those materials. What interests you most?",
                "I'd love to help! I specialize in answering questions about the specific documents that have been shared with me. I can search through them, explain complex topics, provide summaries, and help you understand the content better. What would you like to explore?",
                "I'm your personal assistant for navigating the documents in my knowledge base! I can help you find specific information, answer questions about the content, explain difficult concepts, and discuss any topics covered in those materials. What can I help you discover?",
                "Thanks for asking! I'm here to make the information in my document collection accessible and easy to understand. I can answer questions, provide explanations, find relevant details, and help you explore the topics covered. What would you like to learn about?",
            ]
            return capabilities[hash(query) % len(capabilities)]
            
        elif query_type == 'conversational':
            # Handle different conversational patterns
            query_lower = query.lower()
            
            if any(word in query_lower for word in ['thanks', 'thank you', 'appreciated']):
                thanks_responses = [
                    "You're very welcome! I'm always happy to help. If you have any other questions about the documents, feel free to ask!",
                    "My pleasure! It's great to be able to help you find the information you need. Don't hesitate to ask if you have more questions!",
                    "Absolutely! I'm glad I could assist you. If there's anything else from the documents you'd like to explore, just let me know!",
                    "You're so welcome! I really enjoy helping people discover interesting information. Feel free to ask about anything else!",
                ]
                return thanks_responses[hash(query) % len(thanks_responses)]
                
            elif any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
                goodbye_responses = [
                    "Goodbye! It was wonderful helping you today. Feel free to come back anytime with more questions!",
                    "See you later! I really enjoyed our conversation. Don't hesitate to return if you need help with the documents!",
                    "Take care! It was great chatting with you. I'm always here when you need assistance with your questions!",
                    "Farewell! Thanks for the great conversation. I'll be here whenever you want to explore more information!",
                ]
                return goodbye_responses[hash(query) % len(goodbye_responses)]
                
            else:
                # General positive conversational responses
                general_responses = [
                    "I'm doing well, thank you for asking! I'm here and ready to help you with any questions about the documents. What would you like to know?",
                    "Things are great on my end! I'm excited to help you explore the information in my knowledge base. What interests you?",
                    "I'm fantastic, thanks! I love helping people discover interesting information from documents. What can I help you find today?",
                    "All good here! I'm always energized when I get to help someone learn something new. What would you like to explore?",
                ]
                return general_responses[hash(query) % len(general_responses)]
        
        # Fallback
    def _generate_out_of_scope_response(self, query: str, query_type: str) -> str:
        """Generate appropriate response for out-of-scope questions"""
        
        # Handle creator questions specifically
        creator_patterns = [
            r"who\s+(?:is|are|was|were)\s+(?:the\s+)?creator",
            r"who\s+(?:made|created|built|developed)\s+you",
            r"what\s+(?:is|are)\s+your\s+(?:name|creator|maker)",
        ]
        
        query_lower = query.lower()
        for pattern in creator_patterns:
            if re.search(pattern, query_lower):
                return "I was created by Mapaman! I'm here to help you explore and understand the documents in my knowledge base. Is there anything specific from those documents you'd like to learn about?"
        
        # Friendly out-of-scope responses for factual questions
        responses = [
            "I wish I could help with that, but I don't have information on that topic in my knowledge base. I specialize in the documents that have been uploaded to me. Is there something from those materials you'd like to explore instead?",
            "That's an interesting question, but it's outside my area of expertise! I focus on the specific documents in my knowledge base. What would you like to discover from those materials?",
            "I'd love to help, but I don't have reliable information about that topic. I'm designed to be your guide through the uploaded documents. What aspects of those documents would you like to learn about?",
            "Great question! Unfortunately, that's beyond what I can confidently answer since it's not covered in my document collection. However, I'm really knowledgeable about the materials that have been shared with me. What would you like to explore from those?",
            "I appreciate the question, but I don't have enough information about that topic to give you a good answer. I specialize in the documents in my knowledge base though! Is there anything from those materials you're curious about?",
        ]
        
        # Choose response based on query hash for consistency
        response_index = hash(query) % len(responses)
        return responses[response_index]

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
        answer = ""
        relevance_score = 0.0

        try:
            # Classify the query type first
            query_type = self._classify_query_type(query)
            
            # Handle conversational queries without context retrieval
            if query_type in ['greeting', 'capability', 'conversational']:
                answer = self._generate_conversational_response(query, query_type)
            else:
                # Get context and relevance score for factual queries
                context, relevance_score = self.qa_chain.retrieve_context(query)
                
                # Check if query is out of scope
                if self._is_out_of_scope(query, context, relevance_score, query_type):
                    answer = self._generate_out_of_scope_response(query, query_type)
                else:
                    # Process normally for in-scope queries
                    inputs = {"question": query, "chat_history": formatted_history}
                    
                    if stream_callback:
                        prompt_text = self.qa_chain.prompt.format(
                            context=context, 
                            question=query, 
                            chat_history=formatted_history
                        )
                        for chunk in self.llm_main.stream(prompt_text):
                            if hasattr(chunk, 'text'):
                                token = chunk.text
                                answer += token
                                stream_callback(token)
                    else:
                        result = self.qa_chain.invoke(inputs)
                        answer = result.get("text", "")

        except Exception as e:
            logger.error(f"QA failure for client '{self.client_email}', chatbot '{self.chatbot_id}': {e}")
            
            error_messages = [
                "I'm really sorry, but I ran into a technical glitch while processing your question. Our team has been notified. Could you try rephrasing or asking something else?",
                "Oops! Something went wrong on my end. This happens sometimes, and our team is working to fix these issues. Would you mind trying again in a moment?",
                "I apologize for the trouble! I hit a snag while processing your request. Maybe we could approach your question from a different angle?",
                "Well that's embarrassing! I encountered a technical issue. Our team has been notified, and I'd be happy to try again if you'd like to rephrase your question.",
                "I'm having a moment here - looks like I ran into a technical hiccup. Would you mind trying again? I'd really like to help you with this."
            ]
            
            error_index = hash(str(e)) % len(error_messages)
            answer = error_messages[error_index]
            logger.debug(f"Detailed error: {str(e)}")

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
    
    def get_answer_stream(
        self,
        query: str,
        chat_history: str | List[tuple[str, str]] = '',
        client_email: Optional[str] = None,
        chatbot_id: Optional[str] = None
    ):
        client_changed = client_email and client_email != self.client_email
        chatbot_changed = chatbot_id and chatbot_id != self.chatbot_id

        if client_changed or chatbot_changed:
            self.client_email = client_email or self.client_email
            self.chatbot_id = chatbot_id or self.chatbot_id
            self.initialize_retriever(self.client_email, self.chatbot_id)
            self.setup_rag_chain()

        if not self.retriever or not self.qa_chain:
            if not self.initialize_retriever():
                yield f"Please upload documents for chatbot '{self.chatbot_id}'."
                return
            if not self.setup_rag_chain():
                yield "Could not initialize QA."
                return

        interaction_id = str(uuid.uuid4())
        formatted_history = self._format_chat_history(chat_history)
        answer = ""
        relevance_score = 0.0

        try:
            # Classify the query type first
            query_type = self._classify_query_type(query)
            
            # Handle conversational queries without context retrieval
            if query_type in ['greeting', 'capability', 'conversational']:
                answer = self._generate_conversational_response(query, query_type)
                yield answer
            else:
                context, relevance_score = self.qa_chain.retrieve_context(query)
                
                # Check if query is out of scope
                if self._is_out_of_scope(query, context, relevance_score, query_type):
                    out_of_scope_response = self._generate_out_of_scope_response(query, query_type)
                    answer = out_of_scope_response
                    yield out_of_scope_response
                else:
                    # Stream response for in-scope queries
                    prompt_text = self.qa_chain.prompt.format(
                        context=context,
                        question=query,
                        chat_history=formatted_history
                    )

                    for chunk in self.llm_main.stream(prompt_text):
                        if hasattr(chunk, "text"):
                            token = chunk.text
                            answer += token
                            yield token

        except Exception as e:
            logger.error(f"QA failure for client '{self.client_email}', chatbot '{self.chatbot_id}': {e}")
            error_messages = [
                "I'm really sorry, but I ran into a technical glitch while processing your question.",
                "Oops! Something went wrong on my end. Please try again in a moment.",
                "I hit a snag while processing your request. Maybe we could approach it from a different angle?",
                "Well that's embarrassing! I encountered a technical issue. Please try rephrasing.",
                "Looks like I ran into a technical hiccup. Would you mind trying again?"
            ]
            error_index = hash(str(e)) % len(error_messages)
            yield f"\n{error_messages[error_index]}"
            logger.debug(f"Detailed error: {str(e)}")

        # Log the interaction
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
                inters = [i for i in self.interactions if i.get("client_email") == client_email and i.get("chatbot_id") == chatbot_id]
                feeds = [f for f in self.feedback_store if f.get("client_email") == client_email and f.get("chatbot_id") == chatbot_id]
            else:
                inters = [i for i in self.interactions if i.get("client_email") == client_email]
                feeds = [f for f in self.feedback_store if f.get("client_email") == client_email]
        else:
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