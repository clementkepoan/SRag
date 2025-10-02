import os
from dsrag.chat.chat import create_new_chat_thread, get_chat_thread_response
from dsrag.chat.chat_types import ChatThreadParams, ChatResponseInput
from dsrag.database.chat_thread.basic_db import BasicChatThreadDB
from dsrag.knowledge_base import KnowledgeBase
from dsrag.reranker import NoReranker
from datetime import datetime
from dsrag.embedding import OpenAIEmbedding
from dsrag.reranker import CohereReranker
import gradio as gr

embedding_model = OpenAIEmbedding()
reranker_model = CohereReranker()

# --- Knowledge Base setup ---
kb_id = "database_docs"
kb = KnowledgeBase(kb_id, embedding_model=embedding_model, exists_ok=True, reranker=reranker_model)
chat_db = BasicChatThreadDB()




# --- Chat parameters ---
chat_params = ChatThreadParams(
    kb_ids=[kb_id],
    model="gpt-4o",
    temperature=0.2,
    system_message=f"""
You are a helpful assistant.
Today is {datetime.now().strftime("%Y-%m-%d %H:%M")}.
Always consider this when answering.
Dont output the index or source of your information.
Say I dont know if answer is not provided in the context.
""",
    target_output_length="medium"
)
thread_id = create_new_chat_thread(chat_params, chat_db)

# --- Chat function ---
def chat_with_kb(user_query):
    if not user_query.strip():
        return "Please enter a question."
    response_input = ChatResponseInput(user_input=user_query)
    response = get_chat_thread_response(
        thread_id=thread_id,
        get_response_input=response_input,
        chat_thread_db=chat_db,
        knowledge_bases={kb_id: kb},
        stream=False
    )
    return response["model_response"]["content"]

# --- Upload function (safe filepath mode) ---
def upload_document(file_path):
    if file_path is None:
        return "No file uploaded."

    filename = os.path.basename(file_path)
    ext = os.path.splitext(filename)[1].lower()
    supported = ['.pdf', '.docx', '.txt', '.md']
    if ext not in supported:
        return f"Unsupported file type: {ext}"

    try:
        if ext == '.pdf':
            from dsrag.dsparse.file_parsing.non_vlm_file_parsing import extract_text_from_pdf
            text, _ = extract_text_from_pdf(file_path)
        elif ext == '.docx':
            from dsrag.dsparse.file_parsing.non_vlm_file_parsing import extract_text_from_docx
            text = extract_text_from_docx(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        kb.add_document(doc_id=filename, text=text)
        return f"Uploaded and added '{filename}' to the knowledge base."
    except Exception as e:
        return f"Error uploading '{filename}': {e}"

# --- Delete KB ---
def delete_kb():
    global kb, chat_db, thread_id  # important so we can reassign them

    try:
        # Delete old KB + chat
        kb.delete()
        chat_db.delete_chat_thread(thread_id=thread_id)

        # Recreate KB
        kb = KnowledgeBase(
            kb_id,
            embedding_model=embedding_model,
            exists_ok=True,
            reranker=reranker_model
        )

        # Recreate chat DB + thread
        chat_db = BasicChatThreadDB()
        thread_id = create_new_chat_thread(chat_params, chat_db)

        return "Knowledge base has been reset. A new KB and chat thread are ready."
    except Exception as e:
        return f"Error deleting knowledge base: {e}"


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# SchoolRag Chat Demo\nAsk questions to your knowledge base.")
    with gr.Row():
        with gr.Column():
            chat_box = gr.Textbox(lines=2, placeholder="Ask a question...")
            chat_btn = gr.Button("Ask")
        with gr.Column():
            upload = gr.File(
                label="Upload document",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                type="filepath"  # <--- important
            )
            upload_output = gr.Textbox(label="Upload status")
            del_btn = gr.Button("Reset Knowledge Base", variant="stop")
            del_output = gr.Textbox(label="Delete status")
    response_box = gr.Textbox(label="Response", lines=10)

    chat_btn.click(chat_with_kb, inputs=chat_box, outputs=response_box)
    upload.change(upload_document, inputs=upload, outputs=upload_output)
    del_btn.click(delete_kb, outputs=del_output)

# --- Main ---
if __name__ == "__main__":
    demo.launch(share=True)
