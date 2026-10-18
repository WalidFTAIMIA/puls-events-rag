from pathlib import Path

import pandas as pd

from langchain_core.documents import Document

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "events_filtered.csv"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

print("Chargement des données...")

df = pd.read_csv(INPUT_FILE)

print(f"{len(df)} événements chargés")


# --------------------------------------------------
# Création des documents LangChain
# --------------------------------------------------

documents = []

for _, row in df.iterrows():

    content = f"""
Titre : {row.get('title_fr', '')}

Description :
{row.get('description_fr', '')}

Mots clés :
{row.get('keywords_fr', '')}

Ville :
{row.get('location_city', '')}

Catégorie :
{row.get('category', '')}

Date :
{row.get('firstdate_begin', '')}
"""

    metadata = {
        "uid": str(row.get("uid")),
        "title": str(row.get("title_fr")),
        "city": str(row.get("location_city")),
        "region": str(row.get("location_region")),
        "category": str(row.get("category")),
        "date": str(row.get("firstdate_begin")),
    }

    documents.append(
        Document(
            page_content=content.strip(),
            metadata=metadata
        )
    )

print(f"{len(documents)} documents créés")


# --------------------------------------------------
# Chunking
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"{len(chunks)} chunks générés")


# --------------------------------------------------
# Embeddings
# --------------------------------------------------

print("Chargement du modèle d'embeddings...")

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Génération des embeddings...")


# --------------------------------------------------
# Construction FAISS
# --------------------------------------------------

vectorstore = FAISS.from_documents(
    chunks,
    embeddings_model
)

print("Index FAISS créé")


# --------------------------------------------------
# Sauvegarde
# --------------------------------------------------

VECTORSTORE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

vectorstore.save_local(
    str(VECTORSTORE_DIR)
)

print(f"Index sauvegardé dans : {VECTORSTORE_DIR}")