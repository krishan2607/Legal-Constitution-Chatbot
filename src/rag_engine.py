from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json, os

data_path = "../data/constitution.json"
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

with open(data_path, 'r', encoding='utf-8') as f:
    constitution = json.load(f)

texts, titles = [], []
for article_no, entries in constitution.items():
    for item in entries:
        text = item.get("Chunk Text", "").replace("\u2019", "'").replace("\u201d", '"').strip()
        texts.append(text)
        titles.append(item.get("Title", f"Article {article_no}"))

embeddings = model.encode(texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, os.path.join(model_dir, "constitution_index.faiss"))
np.save(os.path.join(model_dir, "texts.npy"), np.array(texts))
np.save(os.path.join(model_dir, "titles.npy"), np.array(titles))

print("✅ FAISS index built successfully.")
