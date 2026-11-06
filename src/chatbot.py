import os
import time

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS


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
Tu es un assistant spécialisé dans les événements culturels.

Ta mission est de recommander des événements à partir du CONTEXTE fourni.

Règles obligatoires :

1. Utilise uniquement les informations présentes dans le contexte.
2. N'invente jamais de titre, date, lieu, prix ou description.
3. Si des événements correspondent à la question, présente-les directement.
4. Ne dis pas que l'information est absente si le contexte contient des événements pertinents.
5. Si aucun événement ne correspond réellement à la question, réponds exactement :

"Je ne dispose pas de suffisamment d'informations dans les événements indexés."

6. Si un événement est seulement proche de la demande, précise qu'il s'agit d'une correspondance approximative.
7. Pour chaque recommandation, indique si possible :
   - le titre ;
   - la ville ou le lieu ;
   - la date ;
   - une courte description.
8. Ne présente pas un événement musical comme un événement de jazz s'il n'est pas explicitement décrit comme du jazz.
9. Réponds en français de manière claire et concise.
10. Si plusieurs événements pertinents sont trouvés, présente-les sous forme de liste.
11. Privilégie toujours les événements les plus directement liés à la question de l'utilisateur.
12. Ne commence jamais par une phrase indiquant qu'il manque des informations si tu présentes ensuite des événements pertinents.
13. Si les résultats sont seulement proches de la question, indique clairement qu'il s'agit de correspondances approximatives.
"""

USER_PROMPT = """
Question :
{question}

CONTEXTE :

{context}

Répond uniquement avec les informations présentes dans le contexte.
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
# CONSTRUCTION DU CONTEXTE
# ==================================================

def build_context(retrieved_documents):

    chunks = []

    for doc in retrieved_documents:

        metadata = doc.metadata

        chunks.append(
            f"""
Titre : {metadata.get("title", "")}

Ville : {metadata.get("city", "")}

Date : {metadata.get("date", "")}

Contenu :
{doc.page_content[:1000]}
"""
        )

    return "\n\n---\n\n".join(chunks)


# ==================================================
# RAG
# ==================================================

def answer_question_with_metrics(question: str):

    total_start = time.perf_counter()

    retrieval_start = time.perf_counter()
    retrieved_documents = retriever.invoke(question)
    retrieval_time = time.perf_counter() - retrieval_start

    if not retrieved_documents:
        return {
            "answer": (
                "Je ne dispose pas de suffisamment "
                "d'informations dans les événements indexés."
            ),
            "retrieval_time_seconds": retrieval_time,
            "context_time_seconds": 0.0,
            "prompt_time_seconds": 0.0,
            "generation_time_seconds": 0.0,
            "total_time_seconds": time.perf_counter() - total_start,
            "retrieved_documents": retrieved_documents,
        }

    context_start = time.perf_counter()
    context = build_context(retrieved_documents)
    context_time = time.perf_counter() - context_start

    prompt_start = time.perf_counter()
    messages = prompt.format_messages(
        question=question,
        context=context,
    )
    prompt_time = time.perf_counter() - prompt_start

    generation_start = time.perf_counter()
    try:
        response = llm.invoke(messages)
        answer = response.content
    except Exception as error:
        answer = (
            "Erreur lors de l'appel Mistral : "
            f"{error}"
        )
    generation_time = time.perf_counter() - generation_start

    return {
        "answer": answer,
        "retrieval_time_seconds": retrieval_time,
        "context_time_seconds": context_time,
        "prompt_time_seconds": prompt_time,
        "generation_time_seconds": generation_time,
        "total_time_seconds": time.perf_counter() - total_start,
        "retrieved_documents": retrieved_documents,
    }


def answer_question(question: str):
    return answer_question_with_metrics(question)["answer"]


# ==================================================
# INTERFACE CLI
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
            "q",
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