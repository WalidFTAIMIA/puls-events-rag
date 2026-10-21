import os

from dotenv import load_dotenv
from mistralai.client import Mistral


def main():
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente. "
            "Ajoute-la dans un fichier .env à la racine du projet."
        )

    print("Clé Mistral trouvée : True")

    client = Mistral(api_key=api_key)

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {
                "role": "user",
                "content": "Réponds uniquement : Bonjour Walid",
            }
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Mistral a renvoyé une réponse vide.")

    print(content)


if __name__ == "__main__":
    main()