import pandas as pd
import numpy as np

path = "2025-cbs_pc6_2022_vol/pc6_2022_vol.xlsx"
def load_household_summary(path):
    df = pd.read_excel(path, skiprows=5, header=[0, 1])

    df.columns = ['_'.join(filter(None, col)).strip() for col in df.columns]


    df = df.rename(columns={
        df.columns[0]: "postcode6",  
        "Huishouden_Totaal": "total_households",
        "Huishouden_Samenstelling_Eenpersoonshuishouden": "single",
        "Huishouden_Samenstelling_Meepersoons zonder kinderen": "couple_no_kids",
        "Huishouden_Samenstelling_Eenouder": "single_parent",
        "Huishouden_Samenstelling_Tweeouder": "with_kids",
        "Energie. gedurende 2022_Elektra verbruik": "elek_usage",
        "Sociale Zekerheid_Totaal uitkeringen_Personen met WW, Bijstand en/of AO uitkering Beneden AOW-leeftijd": "uitkering"
    })

    cols_needed = [
        "postcode6", "total_households",
        "single", "couple_no_kids", "single_parent", "with_kids",
        "elek_usage", "uitkering"
        ]
    df = df[cols_needed]


    df = df.replace(-99997, np.nan)
    df = df.dropna(subset=["postcode6", "total_households"])

    int_cols = ["total_households", "single", "couple_no_kids", "single_parent", "with_kids", "uitkering"]
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["elek_usage"] = pd.to_numeric(df["elek_usage"], errors="coerce").fillna(2500).astype(float)

    df["pct_single"] = df["single"] / df["total_households"]
    df["pct_with_kids"] = df["with_kids"] / df["total_households"]
    # df["pct_rental"] = df["rental"] / df["total_households"]
    df["pct_lihe"] = df["uitkering"] / df["total_households"]

    return df




