import pandas as pd
import numpy as np

def load_placeholder_data(path="placeholder_cbs_dataset.csv", year=2022):
    df = pd.read_csv(path)

    df = df[df["Year"] == year].copy()

    type_map = {
        "ALONE": "single",
        "COUPLE": "couple_no_kids",
        "FAMILY": "with_kids",
        "OTHER": "nonfamily_group"
    }
    df["household_type"] = df["TYPHH31DEC"].map(type_map)
    #df["household_type"] = df["TYPHH31DEC"].map(type_map).fillna("nonfamily_group")

    data = []

    for _, row in df.iterrows():
        data.append({
            "id": row["Household_ID"],
            "income": row["GESTINKH"],
            "location_code": str(row["GWBCODEJJJJ"]).zfill(8),
            "energielabel": row["Energielabel"],
            "elek_usage": row["ELEK"],
            "elec_return": row["ELEKTERUG"],
            "elec_price": 0.4, # adjusted later
            "lambda_loss_aversion": 1.5, # adjusted later
            "household_type": row["household_type"],
            "lihe": row["LIHE"],
            "lekwi": row["LEKWI"],
            "lihezlek": row["LIHEZLEK"],
            "adopted": bool(row["ZONPV"])
        })

    return data
