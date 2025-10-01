from dsrag.knowledge_base import KnowledgeBase

kb = KnowledgeBase("database_docs")

kb.delete()
print("Knowledge base 'database_docs' deleted.")