import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dsrag.knowledge_base import KnowledgeBase
from dsrag.embedding import OpenAIEmbedding
from dsrag.dsparse.file_parsing.non_vlm_file_parsing import extract_text_from_pdf, extract_text_from_docx
from dsrag.reranker import CohereReranker
document_directory = "./data"
kb_id = "database_docs"



embedding_model = OpenAIEmbedding()
reranker_model = CohereReranker()

kb = KnowledgeBase(kb_id, embedding_model=embedding_model, exists_ok=False, reranker=reranker_model)

for file_name in os.listdir(document_directory):
    file_path = os.path.join(document_directory, file_name)
    try:
        if file_name.endswith('.pdf'):
            text, _ = extract_text_from_pdf(file_path)
        elif file_name.endswith('.docx'):
            text = extract_text_from_docx(file_path)
        elif file_name.endswith('.txt') or file_name.endswith('.md'):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"Unsupported file type: {file_name}")
            continue
        kb.add_document(doc_id=file_name, text=text)
        print(f"Added {file_name} to the knowledge base.")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

