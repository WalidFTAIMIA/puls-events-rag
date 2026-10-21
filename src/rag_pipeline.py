from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Chargement du vectorstore...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

question = "Quels concerts sont prévus à Paris ?"

print(f"\nQuestion : {question}\n")

docs = retriever.invoke(question)

print(f"{len(docs)} documents retrouvés\n")

for i, doc in enumerate(docs, start=1):

    print("=" * 80)
    print(f"DOCUMENT {i}")
    print("=" * 80)

    print(doc.metadata)

    print("\n")

    print(doc.page_content[:1000])

    print("\n")
