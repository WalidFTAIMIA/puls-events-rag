import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot import answer_question_with_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation_questions.csv"
RESULTS_FILE = PROJECT_ROOT / "data" / "evaluation_results.csv"
MIN_REQUEST_INTERVAL = 1.1
API_ERROR_PREFIX = "Erreur lors de l'appel Mistral"
COSINE_THRESHOLD = 0.80
BENCHMARK_SECONDS = {
    "retrieval": 1.0,
    "generation": 5.0,
    "total": 6.0,
}


def compute_similarity(
    model,
    expected_answer,
    generated_answer,
):
    """
    Calcule la similarité cosine
    entre la réponse attendue
    et la réponse générée.
    """

    expected_embedding = model.encode(expected_answer)
    generated_embedding = model.encode(generated_answer)

    score = cosine_similarity(
        [expected_embedding],
        [generated_embedding],
    )[0][0]

    return score


def main():

    print(
        "\n========== ÉVALUATION DU RAG ==========\n"
    )

    df = pd.read_csv(
        EVALUATION_FILE
    )

    required_columns = {
        "question",
        "expected_answer",
        "category",
    }
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {', '.join(sorted(missing_columns))}"
        )
    if df.empty:
        raise ValueError("Le fichier d'évaluation est vide.")

    embedding_model = (
        SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    total_time = 0
    total_similarity = 0
    evaluated_count = 0
    last_request_time = 0.0
    results = []

    for _, row in df.iterrows():

        question = row["question"]

        expected_answer = row[
            "expected_answer"
        ]

        wait_time = MIN_REQUEST_INTERVAL - (
            time.perf_counter() - last_request_time
        )
        if last_request_time and wait_time > 0:
            time.sleep(wait_time)

        start_time = time.perf_counter()

        response_metrics = answer_question_with_metrics(question)
        generated_answer = response_metrics["answer"]

        response_time = (
            time.perf_counter()
            - start_time
        )

        last_request_time = time.perf_counter()

        if generated_answer.lstrip().startswith(API_ERROR_PREFIX):
            print(
                "Évaluation arrêtée : l'API Mistral a renvoyé une erreur."
            )
            break

        similarity = (
            compute_similarity(
                embedding_model,
                expected_answer,
                generated_answer,
            )
        )

        total_time += response_time
        total_similarity += similarity
        evaluated_count += 1
        benchmark_ok = (
            response_metrics["retrieval_time_seconds"]
            <= BENCHMARK_SECONDS["retrieval"]
            and response_metrics["generation_time_seconds"]
            <= BENCHMARK_SECONDS["generation"]
            and response_metrics["total_time_seconds"]
            <= BENCHMARK_SECONDS["total"]
        )
        results.append(
            {
                "question": question,
                "category": row["category"],
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "similarity": float(similarity),
                "cosine_threshold": COSINE_THRESHOLD,
                "cosine_pass": bool(similarity >= COSINE_THRESHOLD),
                "retrieval_time_seconds": response_metrics[
                    "retrieval_time_seconds"
                ],
                "context_time_seconds": response_metrics[
                    "context_time_seconds"
                ],
                "prompt_time_seconds": response_metrics[
                    "prompt_time_seconds"
                ],
                "generation_time_seconds": response_metrics[
                    "generation_time_seconds"
                ],
                "response_time_seconds": response_time,
                "benchmark_ok": benchmark_ok,
                "status": "ok",
            }
        )

        print(
            f"Question : {question}"
        )

        print(
            f"Temps : {response_time:.2f} sec"
        )

        print(
            "Temps détaillés : "
            f"retrieval={response_metrics['retrieval_time_seconds']:.2f}s, "
            f"génération={response_metrics['generation_time_seconds']:.2f}s"
        )

        print(
            f"Similarité : "
            f"{similarity:.3f}"
        )

        print(
            f"Seuil cosine {COSINE_THRESHOLD:.2f} : "
            f"{'PASS' if similarity >= COSINE_THRESHOLD else 'FAIL'}"
        )

        print("\nRéponse attendue :")

        print(expected_answer)

        print("\nRéponse générée :")

        print(generated_answer)

        print(
            "\n"
            + "=" * 80
            + "\n"
        )

    if evaluated_count == 0:
        print("Aucun résultat valide à évaluer.")
        return

    pd.DataFrame(results).to_csv(
        RESULTS_FILE,
        index=False,
    )
    print(f"Résultats sauvegardés : {RESULTS_FILE}")

    average_time = total_time / evaluated_count
    average_similarity = total_similarity / evaluated_count

    print(
        "\n========== RÉSULTATS GLOBAUX ==========\n"
    )

    print(
        f"Temps moyen : "
        f"{average_time:.2f} sec"
    )

    print(
        f"Similarité moyenne : "
        f"{average_similarity:.3f}"
    )

    print(
        f"Score global : "
        f"{average_similarity * 100:.2f}%"
    )

    results_df = pd.DataFrame(results)
    print(
        "Benchmark respecté : "
        f"{int(results_df['benchmark_ok'].sum())}/{len(results_df)}"
    )
    print(
        "Seuil cosine atteint : "
        f"{int(results_df['cosine_pass'].sum())}/{len(results_df)}"
    )


if __name__ == "__main__":
    main()