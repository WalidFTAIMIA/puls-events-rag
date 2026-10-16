from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode(
    "Festival de musique à Paris"
)

print(len(embedding))