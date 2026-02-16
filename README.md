# Personal AI Chatbot with Document Q&A

A fully local, privacy-focused AI chatbot built with **LangChain**, **Ollama**, and **Streamlit**, designed to function as a personal assistant with document understanding and knowledge retrieval.

## 🔧 Features

- **LLM-Powered Chat**  
  Runs completely offline using local models (e.g., `mistral:7b`) via [Ollama](https://ollama.com).

- **Streamlit Interface**  
  A modern web chat UI with avatars, personality configuration, and persistent chat memory.

- **📄 Document Q&A**  
  Upload `.pdf` or `.txt` files and ask questions based on their content using a FAISS-powered retriever.

- **💾 Persistent Chat History**  
  Saves chat sessions to disk and reloads them automatically.

- **Configurable Personality**  
  Customise the assistant’s tone and behavior via a system prompt field in the sidebar.

---

## 🛠 Tech Stack

- Python
  - ![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white)
  - ![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=openai&logoColor=white)
  - ![FAISS](https://img.shields.io/badge/FAISS-0099CC?style=for-the-badge&logo=vector&logoColor=white)
  - ![Ollama Embeddings](https://img.shields.io/badge/Ollama_Embeddings-412991?style=for-the-badge&logo=openai&logoColor=white)
  - ![PyPDFLoader](https://img.shields.io/badge/PyPDFLoader-4B8BBE?style=for-the-badge&logo=readthedocs&logoColor=white)
  - ![TextLoader](https://img.shields.io/badge/TextLoader-888888?style=for-the-badge&logo=readthedocs&logoColor=white)

---

## 🚀 Getting Started

```bash
# Clone and install requirements
pip install -r requirements.txt

# Run the chatbot
streamlit run UIChat.py
```

## 🙏 Credits

This project is based on and inspired by the work of [D-artisan/ai-chatbot](https://github.com/D-artisan/ai-chatbot).

