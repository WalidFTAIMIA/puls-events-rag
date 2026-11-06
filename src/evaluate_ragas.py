import os
import time
import types

from importlib import import_module
from pathlib import Path

import pandas as pd

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI


# ==================================================
# RAGAS IMPORTS
# ==================================================

def load_ragas_components():

    import sys

    # Required workaround for Ragas 0.2.x
    if "langchain_community.chat_models.vertexai" not in sys.modules:

        vertexai_module = types.ModuleType(
            "langchain_community.chat_models.vertexai"
        )

        class ChatVertexAI:
            pass

        vertexai_module.ChatVertexAI = ChatVertexAI

        sys.modules[
            "langchain_community.chat_models.vertexai"
        ] = vertexai_module

    ragas_package = import_module("ragas")

    ragas_embeddings = import_module(
        "ragas.embeddings"
    )

    ragas_llms = import_module(
        "ragas.llms"
    )

    answer_module = import_module(
        "ragas.metrics._answer_relevance"
    )

    recall_module = import_module(
        "ragas.metrics._context_recall"
    )

    faithfulness_module = import_module(
        "ragas.metrics._faithfulness"
    )

    dataset_module = import_module(
        "datasets"
    )

    return (
        dataset_module.Dataset,
        ragas_package.evaluate,
        ragas_embeddings.LangchainEmbeddingsWrapper,
        ragas_llms.LangchainLLMWrapper,
        answer_module.answer_relevancy,
        recall_module.context_recall,
        faithfulness_module.faithfulness,
    )


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation_questions.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation_results_ragas.csv"
)

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "vectorstore"
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "ministral-3b-2512"

BENCHMARK_SECONDS = {
    "retrieval": 1.0,
    "generation": 5.0,
    "total": 6.0,
}


# ==================================================
# PROMPT
# ==================================================

def build_prompt(question, context):

    return f"""
Tu es un assistant spécialisé dans les événements culturels.

Utilise uniquement les informations contenues dans le contexte.

Si l'information n'est pas disponible dans le contexte,
réponds :

"Je ne dispose pas de suffisamment d'informations dans les événements indexés."

Question :

{question}

Contexte :

{context}
"""


# ==================================================
# MAIN
# ==================================================

def main():

    (
        Dataset,
        evaluate,
        LangchainEmbeddingsWrapper,
        LangchainLLMWrapper,
        answer_relevancy,
        context_recall,
        faithfulness,
    ) = load_ragas_components()

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY absente."
        )

    questions = pd.read_csv(
        QUESTIONS_FILE
    )

    embeddings = (
        HuggingFaceEmbeddings(
            model_name=MODEL_NAME
        )
    )

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = (
        vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )
    )

    llm = ChatMistralAI(
        model=LLM_MODEL,
        temperature=0.1,
        api_key=api_key,
    )

    rows = []

    print(
        "\n========== EVALUATION RAGAS ==========\n"
    )

    for _, row in questions.iterrows():

        question = row["question"]

        ground_truth = row[
            "expected_answer"
        ]

        total_start = (
            time.perf_counter()
        )

        retrieval_start = (
            time.perf_counter()
        )

        documents = (
            retriever.invoke(question)
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        contexts = [
            (
                f"Title: "
                f"{doc.metadata.get('title','')}\n"
                f"City: "
                f"{doc.metadata.get('city','')}\n"
                f"{doc.page_content}"
            )
            for doc in documents
        ]

        context = (
            "\n\n---\n\n".join(contexts)
        )

        generation_start = (
            time.perf_counter()
        )

        try:

            response = llm.invoke(
                build_prompt(
                    question,
                    context,
                )
            )

            answer = str(
                response.content
            )

        except Exception as error:

            print(
                f"Erreur pour '{question}':"
            )

            print(error)

            break

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "retrieval_time_seconds": retrieval_time,
                "generation_time_seconds": generation_time,
                "total_time_seconds": total_time,
                "benchmark_ok": (
                    retrieval_time
                    <= BENCHMARK_SECONDS["retrieval"]
                    and generation_time
                    <= BENCHMARK_SECONDS["generation"]
                    and total_time
                    <= BENCHMARK_SECONDS["total"]
                ),
            }
        )

        print(
            f"✓ {question}"
        )

        time.sleep(1.1)

    dataset = Dataset.from_list(
        rows
    )

    ragas_llm = (
        LangchainLLMWrapper(
            llm
        )
    )

    ragas_embeddings = (
        LangchainEmbeddingsWrapper(
            embeddings
        )
    )

    result = evaluate(
        dataset,
        metrics=[
            answer_relevancy,
            context_recall,
            faithfulness,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    metrics_df = (
        result.to_pandas()
        .reset_index(drop=True)
    )

    timing_df = pd.DataFrame(
        [
            {
                "question": row["question"],
                "retrieval_time_seconds": row[
                    "retrieval_time_seconds"
                ],
                "generation_time_seconds": row[
                    "generation_time_seconds"
                ],
                "total_time_seconds": row[
                    "total_time_seconds"
                ],
                "benchmark_ok": row[
                    "benchmark_ok"
                ],
            }
            for row in rows
        ]
    )

    final_df = pd.concat(
        [timing_df, metrics_df],
        axis=1,
    )

    final_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    scores = metrics_df.mean(
        numeric_only=True
    )

    overall_score = (
        scores.mean()
    )

    print(
        "\n========== SCORES MOYENS ==========\n"
    )

    for metric, value in scores.items():

        print(
            f"{metric}: "
            f"{value:.3f}"
        )

    print(
        f"\nSCORE GLOBAL : "
        f"{overall_score:.2%}"
    )

    print(
        "\n========== BENCHMARK ==========\n"
    )

    print(
        f"Retrieval moyen : "
        f"{timing_df['retrieval_time_seconds'].mean():.2f}s"
    )

    print(
        f"Génération moyenne : "
        f"{timing_df['generation_time_seconds'].mean():.2f}s"
    )

    print(
        f"Temps total moyen : "
        f"{timing_df['total_time_seconds'].mean():.2f}s"
    )

    print(
        f"\nRésultats sauvegardés : "
        f"{RESULTS_FILE}"
    )

    print(
        f"Benchmark respecté : "
        f"{sum(row['benchmark_ok'] for row in rows)}/"
        f"{len(rows)}"
    )


if __name__ == "__main__":
    main()