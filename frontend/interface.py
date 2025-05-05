import gradio as gr
import logging
import re
import os
import uuid

from backend.utils.loader import load_document
from backend.utils.text_processor import split_documents
from backend.utils.vectorStore import (
    create_vectorstore, reset_vectorstore, get_vectorstore,
    list_client_chatbots, update_vectorstore, delete_document, _load_metadata
)
from backend.utils.rag import get_rag_system

logger = logging.getLogger(__name__)

SUPPORTED_FILE_EXTENSIONS = [".pdf", ".txt", ".doc", ".docx"]

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_file_loader(file_path):
    _, ext = os.path.splitext(file_path.lower())
    return load_document if ext in SUPPORTED_FILE_EXTENSIONS else None

def process_uploaded_files(files):
    all_docs, processed, failed = [], 0, 0

    for file in files:
        loader_func = get_file_loader(file.name)
        if not loader_func:
            logger.warning(f"Unsupported file: {file.name}")
            failed += 1
            continue

        try:
            docs = loader_func(file.name)
            if docs:
                all_docs.extend(docs)
                processed += 1
            else:
                logger.warning(f"No content from {file.name}")
                failed += 1
        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")
            failed += 1

    return all_docs, processed, failed

def refresh_chatbots(email_state):
    client_email = email_state.get("client_email")
    if not client_email:
        return []
    try:
        chatbots = list_client_chatbots(client_email)
        return [{"label": bot["name"], "value": bot["id"]} for bot in chatbots]
    except Exception as e:
        logger.error(f"Error fetching chatbots: {e}")
        return []

def handle_file_upload(files, chatbot_name, email_state, chatbot_state):
    if not files:
        return "⚠️ No files uploaded.", [], gr.update()
    
    client_email = email_state.get("client_email")
    if not client_email:
        return "⚠️ Please log in first.", [], gr.update()
    
    chatbot_name = chatbot_name or f"Chatbot {uuid.uuid4().hex[:6]}"
    chatbot_id = f"cb_{uuid.uuid4().hex[:10]}"

    all_docs, processed, failed = process_uploaded_files(files)
    if not all_docs:
        return "⚠️ Failed to extract any content.", [], gr.update()

    # Process each document with its file name
    all_chunks = []
    for doc in all_docs:
        # Get the original file name from the document metadata
        file_name = doc.metadata.get("source", "unknown_file")
        # Split the document and add to the chunks list
        doc_chunks = split_documents([doc], file_name)
        all_chunks.extend(doc_chunks)
    
    create_vectorstore(all_chunks, client_email, chatbot_id, chatbot_name)

    rag_system = get_rag_system(client_email, chatbot_id)
    rag_system.initialize_retriever(client_email, chatbot_id)
    rag_system.setup_rag_chain()

    chatbot_state.update({
        "current_chatbot_id": chatbot_id,
        "current_chatbot_name": chatbot_name
    })

    available_chatbots = refresh_chatbots(email_state)
    if chatbot_id not in [bot["value"] for bot in available_chatbots]:
        available_chatbots.append({"label": chatbot_name, "value": chatbot_id})

    status = f"✅ Created chatbot '{chatbot_name}' ({processed} files, {len(all_chunks)} chunks)"
    if failed:
        status += f"\n⚠️ {failed} file(s) failed."

    return status, available_chatbots, gr.update(choices=[b["value"] for b in available_chatbots], value=chatbot_id)

def add_to_existing_chatbot(files, chatbot_id, email_state):
    if not files:
        return "⚠️ No files uploaded.", []
    client_email = email_state.get("client_email")
    if not client_email:
        return "⚠️ Please log in first.", []
    if not chatbot_id:
        return "⚠️ Select a chatbot first.", []
    if not get_vectorstore(client_email, chatbot_id):
        return f"⚠️ Chatbot ID {chatbot_id} not found.", []

    all_docs, processed, failed = process_uploaded_files(files)
    if not all_docs:
        return "⚠️ Failed to extract any content.", []
    all_chunks = []
    for doc in all_docs:
        file_name = doc.metadata.get("source", "unknown_file")
        doc_chunks = split_documents([doc], file_name)
        all_chunks.extend(doc_chunks)
    update_vectorstore(all_chunks, client_email, chatbot_id)
    status = f"✅ Added {processed} files to chatbot."
    if failed:
        status += f"\n⚠️ {failed} file(s) failed."
    return status, refresh_chatbots(email_state)

def select_chatbot(chatbot_id, email_state, chatbot_state):
    if not chatbot_id:
        return chatbot_state, "⚠️ Please select a chatbot.", []
    client_email = email_state.get("client_email")
    if not client_email:
        return chatbot_state, "⚠️ Please log in first.", []
    chatbots = list_client_chatbots(client_email)
    selected = next((bot for bot in chatbots if bot["id"] == chatbot_id), None)
    if not selected:
        return chatbot_state, f"⚠️ Chatbot ID {chatbot_id} not found.", []

    chatbot_state.update({
        "current_chatbot_id": chatbot_id,
        "current_chatbot_name": selected["name"]
    })
    return chatbot_state, f"✅ Switched to: {selected['name']}", []

def delete_chatbot(chatbot_id, email_state, chatbot_state):
    if not chatbot_id:
        return "⚠️ Select a chatbot to delete.", [], chatbot_state
    client_email = email_state.get("client_email")
    if not client_email:
        return "⚠️ Please log in first.", [], chatbot_state
    if not get_vectorstore(client_email, chatbot_id):
        return f"⚠️ Chatbot ID {chatbot_id} not found.", [], chatbot_state

    try:
        reset_vectorstore(client_email, chatbot_id)
        if chatbot_state.get("current_chatbot_id") == chatbot_id:
            chatbot_state.update({"current_chatbot_id": None, "current_chatbot_name": None})
        return f"✅ Deleted chatbot.", refresh_chatbots(email_state), chatbot_state
    except Exception as e:
        logger.error(f"Error deleting chatbot: {e}")
        return f"❌ Error: {str(e)}", [], chatbot_state

def chat(message, history, chat_state, email_state, chatbot_state):
    if not message.strip():
        return history, history, chat_state, ""
    client_email = email_state.get("client_email")
    chatbot_id = chatbot_state.get("current_chatbot_id")
    if not client_email:
        return history + [{"role": "assistant", "content": "⚠️ Please log in first."}], history, chat_state, ""
    if not chatbot_id:
        return history + [{"role": "assistant", "content": "⚠️ Select or create a chatbot."}], history, chat_state, ""
    if not get_vectorstore(client_email, chatbot_id):
        return history + [{"role": "assistant", "content": "⚠️ This chatbot has no documents."}], history, chat_state, ""

    try:
        rag_system = get_rag_system(client_email, chatbot_id)
        formatted_history = [(item["content"], history[idx + 1]["content"])
                             for idx, item in enumerate(history[:-1])
                             if item["role"] == "user" and idx + 1 < len(history) and history[idx + 1]["role"] == "assistant"]
        result = rag_system.get_answer(message, chat_history=formatted_history,
                                       client_email=client_email, chatbot_id=chatbot_id)
        chat_state["current_interaction_id"] = result.get("interaction_id")
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]}
        ]
        return updated_history, updated_history, chat_state, ""
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return history + [{"role": "assistant", "content": f"❌ Error: {str(e)}"}], history, chat_state, ""
    
def list_documents(email_state, chatbot_state):
    client_email = email_state.get("client_email")
    chatbot_id = chatbot_state.get("current_chatbot_id")
    if not client_email or not chatbot_id:
        return []

    metadata = _load_metadata(client_email, chatbot_id)
    if not metadata or not metadata.document_list:
        return []

    # Create a dictionary to track unique document names
    unique_docs = {}
    
    # Group documents by name
    for doc in metadata.document_list:
        doc_name = doc["name"]
        if doc_name not in unique_docs:
            unique_docs[doc_name] = {
                "id": doc["id"],
                "label": doc_name,
                "value": doc_name,  # We'll use name as the value for consistency
                "original_ids": [doc["id"]]
            }
        else:
            # If we already have this document name, add its ID to the list
            unique_docs[doc_name]["original_ids"].append(doc["id"])
    
    # Convert to the expected format for the dropdown
    return [{"label": info["label"], "value": info["value"]} for info in unique_docs.values()]

def handle_delete_document(document_id, email_state, chatbot_state):
    client_email = email_state.get("client_email")
    chatbot_id = chatbot_state.get("current_chatbot_id")
    if not client_email or not chatbot_id:
        return "⚠️ Please log in and select a chatbot.", []

    try:
        # Handle when document_id is passed as a dict (from dropdown)
        if isinstance(document_id, dict) and "value" in document_id:
            document_id = document_id["value"]
            
        # Get metadata
        metadata = _load_metadata(client_email, chatbot_id)
        if not metadata or not metadata.document_list:
            return "⚠️ No documents found for this chatbot.", []
        
        # Since we're now passing document name as the identifier, find all documents with this name
        if any(doc["name"] == document_id for doc in metadata.document_list):
            # This is a document name, not an ID
            docs_to_delete = [doc for doc in metadata.document_list if doc["name"] == document_id]
            doc_name = document_id  # The name is our identifier
        else:
            # Fallback to ID-based lookup (for compatibility)
            docs_to_delete = [doc for doc in metadata.document_list if doc["id"] == document_id]
            if docs_to_delete:
                doc_name = docs_to_delete[0].get("name", "Unknown document")
            else:
                return f"⚠️ Document '{document_id}' not found.", []
        
        if not docs_to_delete:
            return f"⚠️ Document '{document_id}' not found.", []
            
        # Count total chunks to be removed
        total_chunks = sum(len(doc.get("chunk_ids", [])) for doc in docs_to_delete)
        
        # Delete each document and its chunks
        success = True
        for doc in docs_to_delete:
            # Delete all chunks from this document
            doc_id = doc["id"]
            result = delete_document(client_email, chatbot_id, doc_id)
            if not result:
                success = False
                
        if not success:
            return f"❌ Failed to delete some parts of document: {doc_name}", []
            
        # Get updated document list
        updated_docs = list_documents(email_state, chatbot_state)
        return f"✅ Deleted document: {doc_name} ({total_chunks} chunks removed)", updated_docs
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return f"❌ Error deleting document: {str(e)}", []
    
def reset_chat():
    return [], {"current_interaction_id": None}

def submit_feedback(score, helpful, chat_state, email_state, chatbot_state):
    interaction_id = chat_state.get("current_interaction_id")
    client_email = email_state.get("client_email")
    chatbot_id = chatbot_state.get("current_chatbot_id")
    if not interaction_id or not client_email or not chatbot_id:
        return "⚠️ Missing context for feedback."
    rag_system = get_rag_system(client_email, chatbot_id)
    success = rag_system.add_feedback(interaction_id, score, helpful)
    return "✅ Feedback submitted!" if success else "❌ Failed to record feedback."

def get_system_stats(email_state, chatbot_state):
    client_email = email_state.get("client_email")
    chatbot_id = chatbot_state.get("current_chatbot_id")
    if not client_email or not chatbot_id:
        return "⚠️ Please log in and select a chatbot."

    rag_system = get_rag_system(client_email, chatbot_id)
    stats = rag_system.get_system_stats(client_email, chatbot_id)
    feedback = stats.get("feedback_analysis", {})
    if "message" in feedback:
        return feedback["message"]
    name = chatbot_state.get("current_chatbot_name", chatbot_id)
    return f"""
## 📊 Statistics for '{name}'
- Total interactions: {feedback.get('total_interactions', 0)}
- With feedback: {feedback.get('interactions_with_feedback', 0)}
- Helpful %: {feedback.get('helpful_percentage', 0):.1f}%
- Avg score: {feedback.get('average_score', 0):.1f}/5
- Docs with feedback: {stats.get('documents_with_feedback', 0)}
"""

def email_submit(email, email_state):
    if not email.strip():
        return email_state, "⚠️ Please enter an email.", gr.update(visible=True), gr.update(visible=False)
    if not validate_email(email):
        return email_state, "⚠️ Invalid email.", gr.update(visible=True), gr.update(visible=False)
    email_state["client_email"] = email
    return email_state, f"✅ Logged in as {email}", gr.update(visible=False), gr.update(visible=True)

def create_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Initialize states
        email_state = gr.State({"client_email": None})
        chat_state = gr.State({"current_interaction_id": None})
        chatbot_state = gr.State({"current_chatbot_id": None, "current_chatbot_name": None})
        
        # Email Authentication Modal
        with gr.Group(visible=True) as email_modal:
            gr.Markdown("# 📧 Please enter your email to continue")
            email_input = gr.Textbox(label="Email Address", placeholder="your.email@example.com")
            email_status = gr.Textbox(label="Status", interactive=False)
            email_btn = gr.Button("Submit", variant="primary")
        
        # Main UI (initially hidden)
        with gr.Group(visible=False) as main_ui:
            gr.Markdown("# 🧠 Multi-Chatbot RAG System")
            
            with gr.Row():
                # Left panel for chatbot management
                with gr.Column(scale=1):
                    # Create new chatbot section
                    with gr.Group():
                        gr.Markdown("## 🤖 Create New Chatbot")
                        chatbot_name_input = gr.Textbox(
                            label="New Chatbot Name", 
                            placeholder="My Custom Chatbot"
                        )
                        new_files = gr.File(
                            file_count="multiple", 
                            file_types=[".pdf", ".txt", ".doc", ".docx"],
                            label="Upload Documents"
                        )
                        create_btn = gr.Button("🔨 Create New Chatbot", variant="primary")
                        create_status = gr.Textbox(label="Status", lines=3)
                    
                    # Existing chatbots section
                    with gr.Group():
                        gr.Markdown("## 📋 My Chatbots")
                        chatbot_dropdown = gr.Dropdown(
                            label="Select a Chatbot", 
                            choices=[], 
                            interactive=True,
                            allow_custom_value=True
                        )
                        with gr.Row():
                            select_btn = gr.Button("Select", variant="secondary")
                            delete_btn = gr.Button("Delete", variant="stop")
                        select_status = gr.Textbox(label="Status")
                    
                    # Add documents to existing chatbot
                    with gr.Group():
                        gr.Markdown("## 📄 Add Documents to Selected Chatbot")
                        add_files = gr.File(
                            file_count="multiple", 
                            file_types=[".pdf", ".txt", ".doc", ".docx"],
                            label="Upload Additional Documents"
                        )
                        add_btn = gr.Button("📥 Add to Chatbot", variant="secondary")
                        add_status = gr.Textbox(label="Status", lines=3)

                    # Document management section
                    with gr.Group():
                        gr.Markdown("## 🗑️ Delete Documents from Chatbot")
                        document_dropdown = gr.Dropdown(
                            label="Select Document to Delete",
                            choices=[],
                            interactive=True
                        )
                        delete_doc_btn = gr.Button("Delete Document", variant="stop")
                        delete_doc_status = gr.Textbox(label="Status")

                    # System stats
                    with gr.Accordion("📈 System Information", open=False):
                        stats_btn = gr.Button("View Stats for Current Chatbot")
                        stats_output = gr.Markdown("No statistics yet.")

                # Right panel for chat interface
                with gr.Column(scale=2):
                    gr.Markdown(lambda state: f"## 💬 Chatting with: {state.get('current_chatbot_name', 'No chatbot selected')}", inputs=[chatbot_state])
                    chatbot = gr.Chatbot(height=400, type="messages")
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
1. **Create a new chatbot** with a custom name and upload documents
2. OR **Select an existing chatbot** from the dropdown
3. **Add more documents** to existing chatbots as needed
4. Ask questions about the documents in the selected chatbot
5. Rate the responses and provide feedback to improve the system
""")

        # Email modal actions
        email_btn.click(
            email_submit, 
            inputs=[email_input, email_state], 
            outputs=[email_state, email_status, email_modal, main_ui]
        )

        # Chatbot management actions
        create_btn.click(
            handle_file_upload, 
            inputs=[new_files, chatbot_name_input, email_state, chatbot_state], 
            outputs=[create_status, chatbot_dropdown, chatbot_dropdown]
        )
        
        select_btn.click(
            select_chatbot, 
            inputs=[chatbot_dropdown, email_state, chatbot_state], 
            outputs=[chatbot_state, select_status, chatbot]
        )
        
        delete_btn.click(
            delete_chatbot, 
            inputs=[chatbot_dropdown, email_state, chatbot_state], 
            outputs=[select_status, chatbot_dropdown, chatbot_state] 
        )
        
        add_btn.click(
            add_to_existing_chatbot, 
            inputs=[add_files, chatbot_dropdown, email_state], 
            outputs=[add_status, chatbot_dropdown] 
        )

        # Chat interface actions
        send_btn.click(
            chat, 
            inputs=[message_input, chatbot, chat_state, email_state, chatbot_state], 
            outputs=[chatbot, chatbot, chat_state, message_input]
        )
        
        reset_btn.click(
            reset_chat, 
            outputs=[chatbot, chat_state]
        )
        
        feedback_btn.click(
            submit_feedback, 
            inputs=[score_slider, helpful_check, chat_state, email_state, chatbot_state], 
            outputs=[feedback_output]
        )
        
        stats_btn.click(
            get_system_stats, 
            inputs=[email_state, chatbot_state], 
            outputs=[stats_output]
        )
        
        # Refresh chatbot list on page load
        demo.load(
            refresh_chatbots, 
            inputs=[email_state], 
            outputs=[chatbot_dropdown]
        )

        delete_doc_btn.click(
            handle_delete_document,
            inputs=[document_dropdown, email_state, chatbot_state],
            outputs=[delete_doc_status, document_dropdown]
        )

        # Refresh document list when chatbot is selected
        select_btn.click(
            lambda chatbot_id, email_state, chatbot_state: (
                chatbot_state, 
                f"✅ Switched to: {chatbot_id}", 
                [], 
                gr.update(choices=list_documents(email_state, chatbot_state), value=None)
            ),
            inputs=[chatbot_dropdown, email_state, chatbot_state],
            outputs=[chatbot_state, select_status, chatbot, document_dropdown]
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False, pwa=True)