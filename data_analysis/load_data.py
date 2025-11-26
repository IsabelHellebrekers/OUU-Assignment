import os
import pandas as pd 

def get_data_path(filename: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    return os.path.join(data_dir, filename)

def load_demand_history() -> pd.DataFrame:
    path = get_data_path("Demand_2020_2022.csv")
    df = pd.read_csv(path, sep=";")

    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=True)

    return df

def load_demand_forecast() -> pd.DataFrame:
    path = get_data_path("Demand_forecast_20221209.csv")
    df = pd.read_csv(path, sep=";")

    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=True)

    return df

def load_emission_history() -> pd.DataFrame: 
    path = get_data_path("Emission_2020_2022.csv")
    df = pd.read_csv(path, sep=";")

    df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=True)

    return df

def load_generator_parameters() -> pd.DataFrame:
    path = get_data_path("Parameters_generators_NL.csv")
    df = pd.read_csv(path, sep=";")

    return df