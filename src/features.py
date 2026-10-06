def create_features(df_country, horizon: int = 7):
    df = df_country.copy()

    for lag in range(1, 8):
        df[f"lag_{lag}"] = df["new_cases"].shift(lag)

    df["rolling_mean_7"] = df["new_cases"].rolling(7).mean()

    df["target"] = df["new_cases"].shift(-horizon)

    features = [f"lag_{i}" for i in range(1, 8)]
    features += [
        "rolling_mean_7",
        "new_deaths",
        "people_vaccinated",
        "stringency_index"
    ]

    df_model = df.dropna(subset=features + ["target"]).copy()

    X = df_model[features]
    y = df_model["target"]

    return df_model, X, y, features