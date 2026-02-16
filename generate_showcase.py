import json
import os
import time
import config
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Load Configuration and Models
print("Loading models and config...")
llm = ChatOllama(model=config.MODEL_NAME)
embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL_NAME)

# 2. Load Context Data
print("Loading context data...")
docs = []
context_filepath = "context_data.json"
if os.path.exists(context_filepath):
    with open(context_filepath, "r") as f:
        data = json.load(f)
        for key, value in data.items():
            content = f"{key}: {value}"
            docs.append(Document(page_content=content, metadata={"source": "context_data"}))

# 3. Create Vector Store
print("Indexing context...")
if docs:
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
else:
    retriever = None

# 4. Define User Prompt
user_prompt = "hi there - am thinking about getting a new dog but unsure what to look for - i have two teenage children and they are kind and like going on rambles in the country - and we live close to a park, so that's good. i do work 8am-6pm and the kids are at school in those hours so the dog would be on its own a lot. i also cannot afford much in way of vets bills, so can you consider this factor in selecting a dog please - what advice can you give me"

# 5. Retrieve Context
print(f"Retrieving context (k={config.RETRIEVAL_K})...")
context_text = ""
if retriever:
    retrieved_docs = retriever.invoke(user_prompt)
    if retrieved_docs:
        # Use configurable k
        docs_to_use = retrieved_docs[:config.RETRIEVAL_K]
        print(f"Retrieved {len(docs_to_use)} relevant context items.")
        context_text = "\n\n".join([doc.page_content for doc in docs_to_use])

final_user_input = user_prompt
if context_text:
    final_user_input = f"Context:\n{context_text}\n\nQuestion: {user_prompt}"

messages = [
    SystemMessage(content=config.SYSTEM_MESSAGE),
    HumanMessage(content=final_user_input)
]

# 6. Generate Response with Timing
print("Generating response...")
start_time = time.time()
response = llm.invoke(messages)
end_time = time.time()
runtime = end_time - start_time
print(f"Response generated in {runtime:.2f} seconds.")

# 7. Save to Markdown
output_filename = "showcase.md"
markdown_content = f"""# Infrastructure Showcase: Dog Welfare Chatbot Assistant

**Repository:** [Animal Welfare Charity Chatbot](https://github.com/AnimalWelfareCharityChatbot)

## User Query
> {user_prompt}

## Retrieved Context (RAG)
Top {config.RETRIEVAL_K} items retrieved from `context_data.json`:

```text
{context_text}
```

## AI Response
{response.content}

---
*Response generated in {runtime:.2f} seconds.*
*Model: {config.MODEL_NAME} | Embeddings: {config.EMBEDDING_MODEL_NAME}*
"""

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(markdown_content)

print(f"Showcase generated at: {output_filename}")
