from dsrag.knowledge_base import KnowledgeBase

# Create a knowledge base
kb = KnowledgeBase(kb_id="my_knowledge_base")

# Add documents
kb.add_document(
    doc_id="user_manual",  # Use a meaningful ID if possible
    file_path="data/LongRAG.pdf",
    document_title="User Manual",  # Optional but recommended
    metadata={"type": "manual"}    # Optional metadata

)

from dsrag.knowledge_base import KnowledgeBase
from dsrag.reranker import NoReranker

# Load the knowledge base
kb = KnowledgeBase("my_knowledge_base",reranker=NoReranker())

# You can query with multiple search queries
search_queries = [
    "What is attention?",
    "What are the key findings?"
]

# Get results
results = kb.query(search_queries)
print(f"Found {len(results)} results")
#for segment in results:
    #print(segment['text'])  # or segment['content'] if you prefer

# After retrieving segments
context = "\n\n".join([seg['text'] for seg in results])
prompt = f"Based on the following context, answer the question:\n{context}\n\nQuestion: {search_queries[0]}"
# Send 'prompt' to your LLM API and print the response

from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)           

print(response.choices[0].message.content)