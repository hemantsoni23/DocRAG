# frontend/interface.py
import gradio as gr
import logging

from backend.loader import load_pdf
from backend.text_processor import split_documents
from backend.vectorStore import create_vectorstore, reset_vectorstore, get_vectorstore
from backend.rag import get_rag_system

logger = logging.getLogger(__name__)

def handle_file_upload(files):
    try:
        if not files:
            return "⚠️ No files uploaded. Please select at least one PDF."
        
        reset_vectorstore()
        all_docs = []

        for file in files:
            file_path = file.name
            docs = load_pdf(file_path)
            if docs:
                all_docs.extend(docs)

        if not all_docs:
            return "⚠️ Failed to extract content from uploaded documents."

        chunks = split_documents(all_docs)
        create_vectorstore(chunks)

        rag_system = get_rag_system()
        rag_system.initialize_retriever()
        rag_system.setup_rag_chain()

        return f"✅ Successfully processed {len(all_docs)} documents into {len(chunks)} chunks! Ready to chat."
    
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return "❌ Error while processing documents. Please try again."

def chat(message, history, chat_state):
    if not message.strip():
        return [{"role": "assistant", "content": "⚠️ Please enter a message."}], history, chat_state, ""

    if not get_vectorstore():
        return [{"role": "assistant", "content": "⚠️ Please upload a document first."}], history, chat_state, ""

    try:
        rag_system = get_rag_system()
        formatted_history = [(item["content"], history[idx + 1]["content"])
                             for idx, item in enumerate(history[:-1])
                             if item["role"] == "user" and history[idx+1]["role"] == "assistant"]
        
        result = rag_system.get_answer(message, chat_history=formatted_history)

        interaction_id = result.get("interaction_id")
        chat_state["current_interaction_id"] = interaction_id

        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]}
        ]
        
        return updated_history, updated_history, chat_state, ""

    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return [{"role": "assistant", "content": "❌ An error occurred while generating a response."}], history, chat_state, ""


def reset_chat():
    return [], {"current_interaction_id": None}

def submit_feedback(score, helpful, chat_state):
    interaction_id = chat_state.get("current_interaction_id")
    
    if not interaction_id:
        return "⚠️ No interaction to provide feedback for."

    rag_system = get_rag_system()
    success = rag_system.add_feedback(interaction_id, score, helpful)

    if success:
        return "✅ Feedback submitted! Thank you."
    else:
        return "❌ Failed to record feedback."

def get_system_stats():
    rag_system = get_rag_system()
    stats = rag_system.get_system_stats()
    feedback = stats.get("feedback_analysis", {})

    if "message" in feedback:
        return feedback["message"]

    results = f"""
## 📊 System Statistics

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

def create_demo():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧠 Document RAG System with Feedback")
        gr.Markdown("Upload PDFs, chat about them, and improve the system through your feedback!")

        chat_state = gr.State({"current_interaction_id": None})

        with gr.Row():
            with gr.Column(scale=1):
                file_output = gr.Textbox(label="Status")
                files = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDFs")
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

        upload_btn.click(handle_file_upload, inputs=[files], outputs=[file_output])
        send_btn.click(chat, inputs=[message_input, chatbot, chat_state], outputs=[chatbot, chatbot, chat_state, message_input])
        reset_btn.click(reset_chat, outputs=[chatbot, chat_state])
        feedback_btn.click(submit_feedback, inputs=[score_slider, helpful_check, chat_state], outputs=[feedback_output])
        stats_btn.click(get_system_stats, outputs=[stats_output])

        gr.Markdown("""
---
### ℹ️ How to Use:
1. Upload one or more PDFs
2. Ask any question about the documents
3. Rate the responses and provide feedback
""")
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=False)
