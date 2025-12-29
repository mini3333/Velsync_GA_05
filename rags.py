import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load models once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI(api_key="YOUR_API_KEY")

def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, chunks

def retrieve(query, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)

    return [chunks[i] for i in indices[0]]

def generate_answer(query, context):
    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content
