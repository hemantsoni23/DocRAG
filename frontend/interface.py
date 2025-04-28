# frontend/interface.py
import gradio as gr
import logging
import re
import os

from backend.loader import load_document
from backend.text_processor import split_documents
from backend.vectorStore import create_vectorstore, reset_vectorstore, get_vectorstore, list_client_collections
from backend.rag import get_rag_system

logger = logging.getLogger(__name__)

def validate_email(email):
    """Validate email format using regex pattern."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_file_loader(file_path):
    """Return the appropriate loader based on file extension."""
    _, ext = os.path.splitext(file_path.lower())
    
    # Use our new load_document function that supports multiple formats
    return load_document

def handle_file_upload(files, email_state):
    try:
        if not files:
            return "⚠️ No files uploaded. Please select at least one document."
        
        client_email = email_state.get("client_email")
        if not client_email:
            return "⚠️ Email not set. Please log in first."
        
        # Reset only the current client's vectorstore
        # reset_vectorstore(client_email)
        all_docs = []
        processed_files = 0
        failed_files = 0

        for file in files:
            file_path = file.name
            
            # Get the document loader
            loader_func = get_file_loader(file_path)
            
            if loader_func:
                try:
                    docs = loader_func(file_path)
                    if docs:
                        all_docs.extend(docs)
                        processed_files += 1
                    else:
                        failed_files += 1
                        logger.warning(f"No content extracted from {file_path}")
                except Exception as e:
                    failed_files += 1
                    logger.error(f"Error processing {file_path}: {e}")
            else:
                failed_files += 1
                logger.warning(f"Unsupported file type for {file_path}")

        if not all_docs:
            return "⚠️ Failed to extract content from any of the uploaded documents."

        chunks = split_documents(all_docs)
        
        # Create vectorstore for this specific client
        create_vectorstore(chunks, client_email)

        # Get client-specific RAG system
        rag_system = get_rag_system(client_email)
        rag_system.initialize_retriever(client_email)
        rag_system.setup_rag_chain()

        status_message = f"✅ Successfully processed {processed_files} document(s) into {len(chunks)} chunks for {client_email}! Ready to chat."
        if failed_files > 0:
            status_message += f"\n⚠️ Failed to process {failed_files} file(s)."
            
        return status_message
    
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return f"❌ Error while processing documents: {str(e)[:100]}... Please try again."

def chat(message, history, chat_state, email_state):
    if not message.strip():
        return [{"role": "assistant", "content": "⚠️ Please enter a message."}], history, chat_state, ""
    
    client_email = email_state.get("client_email")
    if not client_email:
        return [{"role": "assistant", "content": "⚠️ Email not set. Please log in first."}], history, chat_state, ""

    # Check if the client has a vectorstore
    if not get_vectorstore(client_email):
        return [{"role": "assistant", "content": "⚠️ Please upload documents first."}], history, chat_state, ""

    try:
        # Get the client-specific RAG system
        rag_system = get_rag_system(client_email)
        
        formatted_history = [(item["content"], history[idx + 1]["content"])
                             for idx, item in enumerate(history[:-1])
                             if item["role"] == "user" and history[idx+1]["role"] == "assistant"]
        
        # Pass client_email to get_answer
        result = rag_system.get_answer(
            message, 
            chat_history=formatted_history,
            client_email=client_email
        )

        interaction_id = result.get("interaction_id")
        chat_state["current_interaction_id"] = interaction_id

        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]}
        ]
        
        return updated_history, updated_history, chat_state, ""

    except Exception as e:
        logger.error(f"Error during chat for client {client_email}: {e}")
        return [{"role": "assistant", "content": f"❌ An error occurred while generating a response: {str(e)[:100]}..."}], history, chat_state, ""

def reset_chat():
    return [], {"current_interaction_id": None}

def submit_feedback(score, helpful, chat_state, email_state):
    interaction_id = chat_state.get("current_interaction_id")
    client_email = email_state.get("client_email")
    
    if not interaction_id:
        return "⚠️ No interaction to provide feedback for."
    
    if not client_email:
        return "⚠️ Email not set. Please log in first."

    # Get client-specific RAG system
    rag_system = get_rag_system(client_email)
    success = rag_system.add_feedback(interaction_id, score, helpful)

    if success:
        return "✅ Feedback submitted! Thank you."
    else:
        return "❌ Failed to record feedback."

def get_system_stats(email_state):
    client_email = email_state.get("client_email")
    if not client_email:
        return "⚠️ Email not set. Please log in first."
    
    # Get client-specific RAG system and stats
    rag_system = get_rag_system(client_email)
    stats = rag_system.get_system_stats(client_email)
    feedback = stats.get("feedback_analysis", {})

    if "message" in feedback:
        return feedback["message"]

    results = f"""
## 📊 System Statistics for {client_email}

**Interactions:**
- Total: {feedback.get('total_interactions', 0)}
- With feedback: {feedback.get('interactions_with_feedback', 0)}

**Quality:**
- Helpful responses: {feedback.get('helpful_percentage', 0):.1f}%
- Average score: {feedback.get('average_score', 0):.1f}/5

**Learning:**
- Documents with feedback: {stats.get('documents_with_feedback', 0)}
"""
    return results

def email_submit(email, email_state):
    """Handle email submission and validation."""
    if not email.strip():
        return email_state, "⚠️ Please enter an email address.", gr.update(visible=True), gr.update(visible=False)
    
    if not validate_email(email):
        return email_state, "⚠️ Please enter a valid email address.", gr.update(visible=True), gr.update(visible=False)
    
    # Store email in state
    email_state["client_email"] = email
    return email_state, f"✅ Welcome! Logged in as {email}", gr.update(visible=False), gr.update(visible=True)

def create_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Initialize states
        email_state = gr.State({"client_email": None})
        chat_state = gr.State({"current_interaction_id": None})
        
        # Email Authentication Modal
        with gr.Group(visible=True) as email_modal:
            gr.Markdown("# 📧 Please enter your email to continue")
            email_input = gr.Textbox(label="Email Address", placeholder="your.email@example.com")
            email_status = gr.Textbox(label="Status", interactive=False)
            email_btn = gr.Button("Submit", variant="primary")
        
        # Main UI (initially hidden)
        with gr.Group(visible=False) as main_ui:
            gr.Markdown("# 🧠 Document RAG System with Feedback")
            gr.Markdown("Upload documents, chat about them, and improve the system through your feedback!")

            with gr.Row():
                with gr.Column(scale=1):
                    file_output = gr.Textbox(label="Status", lines=3)
                    files = gr.File(
                        file_count="multiple", 
                        file_types=[".pdf", ".txt", ".doc", ".docx"],
                        label="Upload Documents"
                    )
                    upload_btn = gr.Button("📄 Process Documents", variant="primary")

                    with gr.Accordion("📈 System Information", open=False):
                        stats_btn = gr.Button("View System Stats")
                        stats_output = gr.Markdown("No statistics yet.")

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=400, label="Chat with your documents", type="messages")
                    message_input = gr.Textbox(placeholder="Ask a question...", scale=7)

                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        reset_btn = gr.Button("Reset Chat", variant="secondary")
                    
                    with gr.Row():
                        score_slider = gr.Slider(1, 5, value=3, step=1, label="Rate Answer (1-5)")
                        helpful_check = gr.Checkbox(label="Was this answer helpful?")

                    feedback_btn = gr.Button("Submit Feedback")
                    feedback_output = gr.Textbox(label="Feedback Status")

            # Show current user email
            with gr.Row():
                gr.Markdown(lambda s: f"**Logged in as:** {s.get('client_email', 'Not logged in')}", inputs=[email_state])

            gr.Markdown("""
---
### ℹ️ How to Use:
1. Upload one or more documents (PDF, TXT, DOC, DOCX)
2. Ask any question about the documents
3. Rate the responses and provide feedback
""")

        # Email modal actions
        email_btn.click(
            email_submit, 
            inputs=[email_input, email_state], 
            outputs=[email_state, email_status, email_modal, main_ui]
        )

        # Main UI actions
        upload_btn.click(handle_file_upload, inputs=[files, email_state], outputs=[file_output])
        send_btn.click(chat, inputs=[message_input, chatbot, chat_state, email_state], outputs=[chatbot, chatbot, chat_state, message_input])
        reset_btn.click(reset_chat, outputs=[chatbot, chat_state])
        feedback_btn.click(submit_feedback, inputs=[score_slider, helpful_check, chat_state, email_state], outputs=[feedback_output])
        stats_btn.click(get_system_stats, inputs=[email_state], outputs=[stats_output])

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)