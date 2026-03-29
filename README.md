# ⚖️ Indian Constitution Legal Assistant (RAG Chatbot)

An AI-powered legal assistant that answers complex queries strictly based on the Indian Constitution. As an AI & Data Science project, this application utilizes a Retrieval-Augmented Generation (RAG) architecture to ensure responses are factually accurate, hallucination-free, and grounded entirely in the provided legal text.

## 🚀 Features
* **Fact-Based Answers:** Uses a custom FAISS vector database to retrieve exact articles and clauses from the Constitution before generating a response.
* **Local LLM Integration:** Powered by the **Phi-3** model via Ollama and LiteLLM, ensuring fast, offline, and private inference.
* **Semantic Search:** Utilizes `sentence-transformers` (`all-MiniLM-L6-v2`) to understand the context and meaning of user queries, rather than just basic keyword matching.
* **Interactive UI:** Features a clean, responsive chat interface built with **Streamlit**.

## 🛠️ Tech Stack
* **Language:** Python
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **LLM Orchestration:** LiteLLM / OpenAI Python Client
* **Local Model Server:** Ollama (running Phi-3)
* **Frontend:** Streamlit

## ⚙️ Installation & Setup

**Note:** To keep this repository lightweight and adhere to best practices, the vector database (`.faiss` and `.npy` files) and the virtual environment are not included. You will need to generate the index locally upon your first setup.

### 1. Clone the repository
```bash
git clone [https://github.com/krishan2607/Legal-Constitution-Chatbot.git](https://github.com/krishan2607/Legal-Constitution-Chatbot.git)
cd Legal-Constitution-Chatbot
```

### 2. Set up a Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build the Vector Database
Before running the chatbot, you must generate the FAISS index from the provided `constitution.json` data.
```bash
python src/rag_engine.py
```
*This script will download the open-source embedding model and create a `models/` directory containing `constitution_index.faiss`, `texts.npy`, and `titles.npy`.*

### 5. Start the Local LLM
Ensure you have Ollama installed on your machine and the Phi-3 model downloaded.
```bash
# Start Ollama with the Phi-3 model
ollama run phi3
```
*(Ensure your local LiteLLM proxy is running on port 4000 if you are routing it separately as configured in `chatbot.py`)*

### 6. Run the Application
Launch the Streamlit web interface:
```bash
streamlit run src/app.py
```

## 🧠 How It Works (RAG Architecture)
1. **Ingestion:** `rag_engine.py` reads the Indian Constitution JSON dataset, chunks the text, and converts it into dense vector embeddings.
2. **Storage:** These embeddings are stored in a local FAISS index for lightning-fast similarity search.
3. **Retrieval:** When a user asks a question, the query is embedded, and FAISS retrieves the top most relevant constitutional articles.
4. **Generation:** The retrieved articles are injected into a strict system prompt, instructing the Phi-3 model to answer the question *only* using the provided context, preventing AI hallucinations.

## 👨‍💻 Author
**Krishan Jakhar**
* [LinkedIn](https://linkedin.com/in/krishanjakhar) 
* [GitHub](https://github.com/krishan2607)
