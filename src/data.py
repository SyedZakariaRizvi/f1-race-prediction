from pathlib import Path
import pandas as pd


def build_master_dataset(data_dir):
    """Merge raw tables and calculate the original historical features."""
    data_dir = Path(data_dir)
    results = pd.read_csv(data_dir / "results.csv", na_values=["\\N"])
    races = pd.read_csv(data_dir / "races.csv", na_values=["\\N"])
    drivers = pd.read_csv(data_dir / "drivers.csv", na_values=["\\N"])
    constructors = pd.read_csv(data_dir / "constructors.csv", na_values=["\\N"])
    circuits = pd.read_csv(data_dir / "circuits.csv", na_values=["\\N"])
    qualifying = pd.read_csv(data_dir / "qualifying.csv", na_values=["\\N"])

    df = results.merge(
        races[["raceId", "year", "round", "circuitId"]], on="raceId", how="left"
    )

    df = df.merge(
        drivers[["driverId", "forename", "surname", "dob", "nationality"]],
        on="driverId",
        how="left",
    )

    df = df.merge(
        constructors[["constructorId", "name", "nationality"]].rename(
            columns={
                "name": "constructor_name",
                "nationality": "constructor_nationality",
            }
        ),
        on="constructorId",
        how="left",
    )

    df = df.merge(
        circuits[["circuitId", "name", "country"]].rename(
            columns={"name": "circuit_name", "country": "circuit_country"}
        ),
        on="circuitId",
        how="left",
    )

    df = df.merge(
        qualifying[
            ["raceId", "driverId", "constructorId", "position", "q1", "q2", "q3"]
        ].rename(columns={"position": "qualifying_position"}),
        on=["raceId", "driverId", "constructorId"],
        how="left",
    )

    df["driver_name"] = df["forename"] + " " + df["surname"]
    df["dob"] = pd.to_datetime(df["dob"])
    df["driver_age"] = df["year"] - df["dob"].dt.year
    df["starting_grid"] = df["grid"]
    df["season"] = df["year"]
    df["race_round"] = df["round"]
    df["circuit"] = df["circuitId"]

    df = df.sort_values(by=["driverId", "year", "round"]).reset_index(drop=True)

    df["career_starts"] = df.groupby("driverId").cumcount()
    df["career_wins"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.eq(1).shift(fill_value=False).cumsum())
    )
    df["career_podiums"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.le(3).shift(fill_value=False).cumsum())
    )
    df["career_points"] = (
        df.groupby("driverId")["points"]
        .transform(lambda x: x.shift(fill_value=0).cumsum())
    )
    df["avg_finish_before_race"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift().expanding().mean())
    )
    df["recent_form"] = (
        df.groupby("driverId")["positionOrder"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )

    df["constructor_starts"] = df.groupby("constructorId").cumcount()
    df["constructor_wins"] = (
        df.groupby("constructorId")["positionOrder"]
        .transform(lambda x: x.eq(1).shift(fill_value=False).cumsum())
    )
    df["constructor_recent_form"] = (
        df.groupby("constructorId")["positionOrder"]
        .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    )

    return df


def clean_dataset(df):
    """Filter to 2006 onward, drop outcome columns, and fill missing history."""
    df_modern = df[df["season"] >= 2006].copy()

    leaky_cols = [
        "position", "positionText", "points", "laps", "time",
        "milliseconds", "fastestLap", "rank", "fastestLapTime",
        "fastestLapSpeed", "statusId", "resultId"
    ]

    redundant_text_cols = [
        "q1", "q2", "q3", "driver_name", "constructor_name",
        "circuit_name", "forename", "surname", "dob"
    ]

    df_clean = df_modern.drop(columns=leaky_cols + redundant_text_cols)

    df_clean["qualifying_position"] = df_clean["qualifying_position"].fillna(df_clean["starting_grid"])

    rookie_cols = ["avg_finish_before_race", "recent_form", "constructor_recent_form"]
    df_clean[rookie_cols] = df_clean[rookie_cols].fillna(20.0)

    df_clean = df_clean.drop(columns=["grid", "year", "round", "circuit", "number"], errors="ignore")
    return df_clean


def group_rare_categories(df, columns, top_n=10):
    """
    Keeps the top_n most frequent categories in a column and
    renames all remaining rare categories to 'Other'.
    """
    df_grouped = df.copy()

    for col in columns:
        top_categories = df_grouped[col].value_counts().nlargest(top_n).index

        df_grouped[col] = df_grouped[col].where(df_grouped[col].isin(top_categories), 'Other')

    return df_grouped


def split_dataset(df_clean):
    """Map grid zero to 21 and create the original chronological train/test split."""
    df_clean = df_clean.copy()
    df_clean['starting_grid'] = df_clean['starting_grid'].replace(0, 21)

    train_df = df_clean[df_clean['season'] <= 2023]
    test_df = df_clean[df_clean['season'] == 2024]

    X_train = train_df.drop(columns=['positionOrder', 'raceId', 'driverId', 'constructorId', 'circuitId'])
    y_train = train_df['positionOrder']

    X_test = test_df.drop(columns=['positionOrder', 'raceId', 'driverId', 'constructorId', 'circuitId'])
    y_test = test_df['positionOrder']
    return train_df, test_df, X_train, X_test, y_train, y_test
