# Puls-Events RAG

## Contexte

Puls-Events souhaite proposer une plateforme capable de recommander des événements culturels en fonction des préférences des utilisateurs, de leur localisation et de la période recherchée.

L'objectif de ce projet est de réaliser un Proof of Concept (POC) d'un chatbot utilisant une architecture RAG (Retrieval-Augmented Generation). Le chatbot devra retrouver les événements les plus pertinents dans une base vectorielle, puis générer une réponse avec le modèle Mistral.

La source de données retenue est le jeu de données public [Événements publics OpenAgenda](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda).

## Objectifs du projet

Le POC final devra permettre de :

- récupérer les événements depuis OpenAgenda ;
- limiter les données à un périmètre géographique défini ;
- conserver uniquement les événements récents, âgés de moins d'un an ;
- nettoyer et structurer les données ;
- générer des embeddings pour les événements ;
- construire et reconstruire une base vectorielle FAISS ;
- rechercher les événements avec LangChain ;
- générer des réponses avec Mistral ;
- évaluer la qualité des réponses à partir d'un jeu de questions-réponses annoté.

## Avancement

### Étape 1 - Configuration de l'environnement : terminée

L'environnement Python `.venv` a été créé sous Windows. Les principales bibliothèques du projet ont été installées :

- Pandas ;
- Requests ;
- LangChain ;
- FAISS CPU ;
- Mistral ;
- Sentence Transformers pour la génération des embeddings.

Pour utiliser le projet dans VS Code, sélectionner l'interpréteur :

```text
.venv\Scripts\python.exe
```

Installation de la dépendance d'embedding si nécessaire :

```powershell
.\.venv\Scripts\python.exe -m pip install sentence-transformers
```

### Étape 2 - Extraction et préparation des données : terminée

Le périmètre géographique retenu pour le POC est l'Île-de-France.

Le script [`src/extract.py`](src/extract.py) récupère les événements depuis l'API OpenDataSoft qui expose le jeu de données OpenAgenda. Les événements sont filtrés sur la région Île-de-France et enregistrés dans :

```text
data/events_raw.csv
```

Le script [`src/preprocess.py`](src/preprocess.py) :

- sélectionne les colonnes utiles ;
- supprime les événements sans titre ;
- convertit les dates au format datetime ;
- supprime les dates invalides ;
- conserve les événements dont la date est postérieure à la date d'il y a un an ;
- produit le fichier nettoyé `data/events_filtered.csv`.

Pour exécuter les deux étapes depuis la racine du projet :

```powershell
.\.venv\Scripts\python.exe src/extract.py
.\.venv\Scripts\python.exe src/preprocess.py
```

### Étape 3 - Vectorisation et indexation FAISS : terminée

Le script [`src/build_vectorstore.py`](src/build_vectorstore.py) :

- charge les événements nettoyés ;
- construit les documents LangChain ;
- découpe les textes en chunks de 500 caractères avec un chevauchement de 50 caractères ;
- génère les embeddings avec `sentence-transformers/all-MiniLM-L6-v2` ;
- crée un index FAISS ;
- conserve les métadonnées des événements dans l'index LangChain.

Les artefacts sont enregistrés dans le dossier `vectorstore/` :

```text
vectorstore/index.faiss
vectorstore/index.pkl
```

Pour reconstruire l'index :

```powershell
.\.venv\Scripts\python.exe src/build_vectorstore.py
```

La recherche FAISS et les métadonnées sont vérifiées par [`tests/test_vectorstore.py`](tests/test_vectorstore.py).

### Étape 4 - Chatbot RAG : validée

Le script [`src/chatbot.py`](src/chatbot.py) :

- charge l'index FAISS ;
- recherche les cinq chunks les plus proches de la question ;
- construit un contexte avec le titre, la ville, la date et le contenu ;
- utilise LangChain pour construire le prompt ;
- envoie le contexte au modèle Mistral `ministral-3b-2512` ;
- répond dans une interface console sans conserver d'historique.

Le test [`tests/test_mistral.py`](tests/test_mistral.py) vérifie séparément la connexion à l'API Mistral. La génération de réponses a été validée avec le modèle `ministral-3b-2512`.

## Installation

### PowerShell sous Windows

Depuis la racine du projet :

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### WSL

Dans WSL, utilise l'environnement Linux et ses commandes propres :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ne mélange pas l'environnement Windows avec celui de WSL. Pour ce projet, les commandes doivent être exécutées avec le même environnement que celui sélectionné dans VS Code.

## Configuration Mistral

Crée un fichier `.env` à la racine du projet :

```text
MISTRAL_API_KEY=ta_cle_api_mistral
```

Le fichier `.env` est ignoré par Git. Ne publie jamais la clé API dans le dépôt.

Pour vérifier uniquement la présence de la clé sans l'afficher :

```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('MISTRAL_API_KEY')))"
```

Pour lancer le chatbot :

```powershell
.\.venv\Scripts\python.exe src\chatbot.py
```

ou dans WSL :

```bash
python src/chatbot.py
```

Tape `quit`, `exit` ou `q` pour arrêter le chatbot.

### Limite API Mistral

Une erreur HTTP `429 Rate limit exceeded` signifie que la requête est refusée par la limite du compte ou du workspace Mistral. La recherche FAISS peut fonctionner même si l'appel Mistral est bloqué. Vérifie les menus **Usage** et **Limites** dans la console Mistral. Les requêtes refusées peuvent ne pas apparaître dans les statistiques d'utilisation.

## Tests

Pour exécuter les tests locaux :

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Les tests couvrent les événements filtrés, les embeddings, l'index FAISS et les métadonnées. Le test Mistral est un smoke test réseau et doit être lancé séparément :

```powershell
.\.venv\Scripts\python.exe tests\test_mistral.py
```

Il nécessite une clé API valide et peut échouer avec `429` si la limite Mistral est atteinte.

## Évaluation du RAG

Le script [`src/evaluate_rag.py`](src/evaluate_rag.py) compare les réponses générées avec les réponses annotées de [`data/evaluation_questions.csv`](data/evaluation_questions.csv). Il calcule une similarité cosinus avec le modèle `all-MiniLM-L6-v2` et sauvegarde les résultats dans :

```text
data/evaluation_results.csv
```

Pour lancer l'évaluation :

```powershell
.\.venv\Scripts\python.exe src\evaluate_rag.py
```

L'évaluation dépend de l'API Mistral et peut être interrompue par une limite de débit. Le score de similarité est un indicateur du POC, pas une mesure parfaite de la qualité des réponses.

L'intégration Ragas est mise en pause pour le moment afin de préserver la stabilité de l'environnement principal. L'environnement expérimental `.venv-ragas` est séparé et n'est pas nécessaire pour exécuter le chatbot.


## Architecture cible

```text
OpenAgenda / OpenDataSoft
            |
            v
    src/extract.py
            |
            v
    data/events_raw.csv
            |
            v
   src/preprocess.py
            |
            v
  data/events_filtered.csv
            |
            v
       Embeddings
            |
            v
        FAISS
            |
            v
       LangChain
            |
            v
         Mistral
            |
            v
        Chatbot
```

## Structure du projet

```text
puls-events-rag/
├── data/
│   ├── evenements-publics-openagenda.csv
│   ├── events_raw.csv
│   └── events_filtered.csv
├── notebooks/
│   └── exploration_openagenda.ipynb
├── src/
│   ├── extract.py
│   ├── preprocess.py
│   ├── build_vectorstore.py
│   ├── rag_pipeline.py
│   ├── chatbot.py
│   └── evaluate_rag.py
├── tests/
│   ├── test_embedding.py
│   ├── test_events.py
│   ├── test_install.py
│   ├── test_mistral.py
│   └── test_vectorstore.py
├── vectorstore/
├── docs/
│   ├── rapport_technique_puls_events.docx
│   └── presentation_puls_events_rag.pptx
├── requirements.txt
└── README.md
```

## Vérification de l'installation

```powershell
.\.venv\Scripts\python.exe tests/test_install.py
.\.venv\Scripts\python.exe tests/test_embedding.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Données

Les fichiers de données ne sont pas versionnés dans GitHub en raison de leur taille.

Pour reconstruire les données :

```bash
python src/extract.py

python src/preprocess.py
python src/build_vectorstore.py
```

Les fichiers suivants sont générés localement et peuvent être reconstruits :

- `data/events_raw.csv` ;
- `data/events_filtered.csv` ;
- `vectorstore/index.faiss` ;
- `vectorstore/index.pkl`.

## Avancement global

- Étape 1 - Environnement : terminée.
- Étape 2 - Extraction et préparation : validée.
- Étape 3 - Embeddings et index FAISS : validée.
- Étape 4 - Chatbot RAG avec Mistral : validée.
- Évaluation cosine : disponible avec `evaluate_rag.py`.
- Ragas : mis en pause pour le moment.


