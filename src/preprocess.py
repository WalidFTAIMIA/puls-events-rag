import pandas as pd

INPUT_FILE = "data/events_raw.csv"
OUTPUT_FILE = "data/events_filtered.csv"

print("Chargement des données...")

df = pd.read_csv(INPUT_FILE)

print(f"Nombre de lignes initial : {len(df)}")

KEEP_COLUMNS = [
    "uid",
    "title_fr",
    "description_fr",
    "keywords_fr",
    "firstdate_begin",
    "location_name",
    "location_address",
    "location_city",
    "location_department",
    "location_region",
    "category"
]

df = df[KEEP_COLUMNS]

# Suppression des lignes sans titre

df = df.dropna(subset=["title_fr"])

# Conversion des dates

df["firstdate_begin"] = pd.to_datetime(
    df["firstdate_begin"],
    utc=True,
    errors="coerce"
)

# Suppression des dates invalides

df = df.dropna(subset=["firstdate_begin"])

# Moins d'un an

today = pd.Timestamp.now(tz="UTC")

one_year_ago = today - pd.DateOffset(years=1)

df = df[
    df["firstdate_begin"] >= one_year_ago
]

print(
    f"Nombre après filtre temporel : {len(df)}"
)

# Sauvegarde

df = df.fillna("")
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Fichier sauvegardé : {OUTPUT_FILE}"
)
