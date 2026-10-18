import pandas as pd

DATA_PATH = "data/events_filtered.csv"


def test_region_idf():
    """
    Vérifie que tous les événements
    appartiennent à l'Île-de-France.
    """

    df = pd.read_csv(DATA_PATH)

    assert (
        df["location_region"] == "Île-de-France"
    ).all()


def test_events_less_than_one_year():
    """
    Vérifie que tous les événements
    ont moins d'un an.
    """

    df = pd.read_csv(DATA_PATH)

    df["firstdate_begin"] = pd.to_datetime(
        df["firstdate_begin"],
        utc=True
    )

    one_year_ago = (
        pd.Timestamp.now(tz="UTC")
        - pd.DateOffset(years=1)
    )

    assert (
        df["firstdate_begin"] >= one_year_ago
    ).all()


def test_no_missing_title():
    """
    Vérifie que chaque événement
    possède un titre.
    """

    df = pd.read_csv(DATA_PATH)

    assert df["title_fr"].notna().all()


def test_dataset_not_empty():
    """
    Vérifie que le dataset n'est pas vide.
    """

    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0