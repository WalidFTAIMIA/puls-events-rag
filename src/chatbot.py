import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from mistralai.client import Mistral


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise ValueError(
        "MISTRAL_API_KEY introuvable dans le fichier .env"
    )

VECTORSTORE_PATH = "vectorstore"


# --------------------------------------------------
# Chargement des embeddings
# --------------------------------------------------

print("Chargement du modèle d'embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# Chargement de FAISS
# --------------------------------------------------

print("Chargement du vectorstore FAISS...")

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)

print("Vectorstore chargé")


# --------------------------------------------------
# Client Mistral
# --------------------------------------------------

print("Connexion à Mistral...")
client = Mistral(
    api_key=API_KEY
)
print("Client Mistral initialisé")

# --------------------------------------------------
# Construction du contexte
# --------------------------------------------------

def build_context(documents):
    context = []

    for doc in documents:

        title = doc.metadata.get("title", "")
        city = doc.metadata.get("city", "")
        date = doc.metadata.get("date", "")

        context.append(
            f"""
Titre : {title}
Ville : {city}
Date : {date}

{doc.page_content}
"""
        )

    return "\n\n".join(context)


# --------------------------------------------------
# Recherche RAG
# --------------------------------------------------

def answer_question(question: str):

    documents = retriever.invoke(question)

    context = build_context(documents)

    prompt = f"""
Tu es un assistant spécialisé dans les événements culturels.

Réponds uniquement à partir du contexte fourni.

Si l'information n'est pas présente dans le contexte,
réponds que tu ne disposes pas de suffisamment d'informations.

Contexte :
{context}

Question :
{question}
"""

    try:

        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        print("\nFAISS a retrouvé les événements suivants :\n")

        for doc in documents:

            print(
                f"- {doc.metadata.get('title', '')}"
                f" | {doc.metadata.get('city', '')}"
                f" | {doc.metadata.get('date', '')}"
            )

        return (
            "\nErreur lors de l'appel Mistral :\n"
            f"{str(e)}"
        )


# --------------------------------------------------
# Interface console
# --------------------------------------------------

def main():

    print("\n")
    print("=" * 60)
    print("CHATBOT RAG OPENAGENDA")
    print("=" * 60)
    print("Tape 'quit' pour quitter.\n")

    while True:

        question = input("Question : ")

        if question.lower() in [
            "quit",
            "exit",
            "q"
        ]:
            break

        print("\nRecherche...\n")

        answer = answer_question(question)

        print("Réponse :\n")
        print(answer)

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()
