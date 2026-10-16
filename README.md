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
│   └── preprocess.py
├── tests/
│   ├── test_embedding.py
│   └── test_install.py
├── vectorstore/
├── requirements.txt
└── README.md
```

## Vérification de l'installation

```powershell
.\.venv\Scripts\python.exe tests/test_install.py
.\.venv\Scripts\python.exe tests/test_embedding.py
```

## Données

Les fichiers de données ne sont pas versionnés dans GitHub en raison de leur taille.

Pour reconstruire les données :

```bash
python src/extract.py

python src/preprocess.py
