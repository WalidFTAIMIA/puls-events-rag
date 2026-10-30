import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY introuvable dans le fichier .env"
    )


# ==================================================
# PROMPTS
# ==================================================

SYSTEM_PROMPT = """
Tu es un assistant spécialisé dans
les événements culturels.

Utilise uniquement le contexte fourni.

Si l'information n'est pas présente
dans le contexte, réponds :

"Je ne dispose pas de suffisamment
d'informations dans les événements indexés."

N'invente jamais :
- événement
- lieu
- date
- prix
- lien
"""

USER_PROMPT = """
Question :
{question}

CONTEXTE :

{context}

Répond uniquement avec les informations
présentes dans le contexte.
"""


# ==================================================
# EMBEDDINGS
# ==================================================

print("Chargement du modèle d'embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# FAISS
# ==================================================

print("Chargement du vectorstore FAISS...")

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)

print("Vectorstore chargé")


# ==================================================
# MISTRAL
# ==================================================

print("Connexion à Mistral...")

llm = ChatMistralAI(
    model="ministral-3b-2512",
    temperature=0.1,
    api_key=API_KEY,
)

print("Client Mistral initialisé")


# ==================================================
# PROMPT TEMPLATE
# ==================================================

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ]
)


# ==================================================
# CONTEXTE
# ==================================================

def build_context(retrieved_documents):

    chunks = []

    for doc in retrieved_documents:

        meta = doc.metadata

        chunks.append(
            f"""
Titre : {meta.get("title", "")}

Ville : {meta.get("city", "")}

Date : {meta.get("date", "")}

Contenu :
{doc.page_content[:1000]}
"""
        )

    return "\n\n---\n\n".join(chunks)


# ==================================================
# RAG
# ==================================================

def answer_question(question: str):

    retrieved_documents = retriever.invoke(question)

    if not retrieved_documents:
        return (
            "Je ne dispose pas de suffisamment "
            "d'informations dans les événements indexés."
        )

    context = build_context(retrieved_documents)

    try:

        messages = prompt.format_messages(
            question=question,
            context=context,
        )
        response = llm.invoke(messages)

        return response.content

    except Exception as error:  # noqa: BLE001 - l'appel API doit rester non bloquant

        print("\nFAISS a retrouvé :\n")

        for doc in retrieved_documents:

            print(
                f"- {doc.metadata.get('title', '')}"
                f" | {doc.metadata.get('city', '')}"
                f" | {doc.metadata.get('date', '')}"
            )

        return (
            "\nErreur lors de l'appel Mistral :\n"
            f"{error}"
        )


# ==================================================
# CLI
# ==================================================

def main():

    print("\n")
    print("=" * 60)
    print("CHATBOT RAG OPENAGENDA")
    print("=" * 60)
    print("Tape 'quit' pour quitter.\n")

    while True:

        question = input("Question : ").strip()

        if question.lower() in {

            "quit",
            "exit",
            "q"

        }:
            print("Au revoir !")
            break

        if not question:
            continue

        print("\nRecherche...\n")

        answer = answer_question(question)

        print("Réponse :\n")

        print(answer)

        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()