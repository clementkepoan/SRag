from dsrag.create_kb import create_kb_from_directory


dir = "./data"

kb = create_kb_from_directory(kb_id="database_system", directory=dir, title="Computer Science Documents", description="A collection of computer science documents.", language="en")

