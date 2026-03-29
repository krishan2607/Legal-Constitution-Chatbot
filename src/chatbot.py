from openai import OpenAI
import faiss, numpy as np
import os 
from sentence_transformers import SentenceTransformer
import re

# Connect to local LLM (Phi-3 via LiteLLM proxy)
client = OpenAI(base_url="http://localhost:4000", api_key="none")

# Load FAISS + embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

index = faiss.read_index(os.path.join(MODEL_DIR, "constitution_index.faiss"))
texts = np.load(os.path.join(MODEL_DIR, "texts.npy"), allow_pickle=True)
titles = np.load(os.path.join(MODEL_DIR, "titles.npy"), allow_pickle=True)



def retrieve_articles(query, top_k=3):
    match = re.search(r"article\s+(\d+)", query.lower())
    if match:
        num = match.group(1)
        for i, title in enumerate(titles):
            if title.lower().startswith(f"article {num}"):
                return [(titles[i], texts[i])]

    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    return [(titles[i], texts[i]) for i in indices[0]]


def chat_with_constitution(query):
    articles = retrieve_articles(query)

    if not articles:
        return "⚠ Information not found in dataset."

    context = "\n\n".join([f"{a}:\n{t}" for a, t in articles])

    prompt = f"""
    You are an Indian Constitution AI assistant. You must answer strictly using the context below.
    Do not create new information.

    Context:
    {context}

    Question: {query}
    """

    response = client.chat.completions.create(
        model="ollama/phi3:instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("⚖️ Indian Constitution Legal Assistant Chatbot")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("👤 You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        print("\n🤖 Bot:", chat_with_constitution(query))
