import pandas as pd


def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, usecols=["location", "date", "new_cases", "new_deaths", "people_vaccinated", "stringency_index"])
    return df


def preprocess_country_data(
    df: pd.DataFrame,
    country: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    df_country = df[df["location"] == country].copy()

    df_country["date"] = pd.to_datetime(df_country["date"])
    df_country = df_country.sort_values("date")

    df_country["new_cases"] = df_country["new_cases"].fillna(0).clip(lower=0)

    context_cols = ["new_deaths", "people_vaccinated", "stringency_index"]
    df_country[context_cols] = df_country[context_cols].fillna(0)

    df_country = df_country[
        (df_country["date"] >= start_date) &
        (df_country["date"] <= end_date)
    ].copy()

    return df_country
