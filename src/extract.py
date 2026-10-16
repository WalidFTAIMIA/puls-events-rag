import requests
import pandas as pd

URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)

LIMIT = 100
MAX_RECORDS = 10000

offset = 0
all_events = []

print("Début de l'extraction OpenAgenda...")

while len(all_events) < MAX_RECORDS:

    response = requests.get(
        URL,
        params={
            "limit": LIMIT,
            "offset": offset,
            "where": "location_region='Île-de-France'"
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    results = data.get("results", [])

    if not results:
        break

    all_events.extend(results)

    print(
        f"Événements récupérés : {len(all_events)}"
    )

    offset += LIMIT

df = pd.DataFrame(all_events)

output_file = "data/events_raw.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nExtraction terminée")
print(f"Nombre total d'événements : {len(df)}")
print(f"Fichier créé : {output_file}")