import pandas as pd
import numpy as np
import random
from config import n_agents


path = "Processed_ams_pc6_buurt_data.csv"
income_df = pd.read_csv("Distribution of standardised income, 2022.csv")

income_df = pd.read_csv("Distribution of standardised income, 2022.csv")

income_df = income_df.rename(columns={
    "standardised income (x 1000 euros)": "income_bin",
    "Single persion": "single",
    "Couple without children": "couple_no_kids",
    "Couple with children": "with_kids",
    "Single-parent family": "single_parent"
})

def parse_income(bin_label):
    if "less than" in bin_label:
        return -7000
    elif "more than" in bin_label:
        return 130000
    else:
        parts = bin_label.replace("between ", "").split(" and ")
        return int((float(parts[0]) + float(parts[1])) * 1000 / 2)

income_df["income_value"] = income_df["income_bin"].apply(parse_income)

def get_income_sampler(df, col):
    values = df["income_value"]
    weights = df[col].fillna(0).astype(float)
    probs = weights / weights.sum()
    return lambda: np.random.choice(values, p=probs)

income_samplers = {
    "single": get_income_sampler(income_df, "single"),
    "couple_no_kids": get_income_sampler(income_df, "couple_no_kids"),
    "with_kids": get_income_sampler(income_df, "with_kids"),
    "single_parent": get_income_sampler(income_df, "single_parent")
}

def sample_income(hh_type):
    return income_samplers.get(hh_type, income_samplers["single"])()


def dynamic_elec_price(timestep):
    base_price = 0.45  # 2022
    min_price = 0.25   
    decline_per_year = 0.03  

    price = base_price - decline_per_year * timestep
    return max(min_price, price)

def get_income_quantiles(agents):
    income_list = [agent["income"] for agent in agents]
    q33 = np.percentile(income_list, 33)
    q66 = np.percentile(income_list, 66)
    return (q33, q66)


def load_amsterdam_data(max_agents=n_agents):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["postcode6", "buurt_code"])

    agents = []
    agent_id = 0

    for _, row in df.iterrows():
        total = int(row["total_households"])
        postcode6 = row["postcode6"]
        buurt_code = row["buurt_code"]
        elek_usage_total = row["elek_usage"]

        hh_types = {
            "single": int(row["single"]),
            "couple_no_kids": int(row["couple_no_kids"]),
            "single_parent": int(row["single_parent"]),
            "with_kids": int(row["with_kids"])
        }

        vulnerable_count = int(row["vulnerable"])
        vulnerable_flags = [1]*vulnerable_count + [0]*(total - vulnerable_count)
        random.shuffle(vulnerable_flags)

        elek_usage_hh = np.random.dirichlet(np.ones(total)) * elek_usage_total

        all_types = []
        for t, n in hh_types.items():
            all_types += [t] * n
        random.shuffle(all_types)

        for i in range(total):
            hh_type = all_types[i % len(all_types)]
            income = max(10000, sample_income(hh_type))
            energielabel = random.choices(list("ABCDEFG"), weights=[0.15, 0.2, 0.25, 0.2, 0.1, 0.07, 0.03])[0]
            adopted = random.random() < 0.2

            agent = {
                "id": agent_id,
                "income": income,
                "energielabel": energielabel,
                "elek_usage": elek_usage_hh[i],
                "elec_price": dynamic_elec_price(timestep=0),  
                "household_type": hh_type,
                "postcode6": postcode6,
                "buurt_code": buurt_code,
                "lihe": vulnerable_flags[i],
                "adopted": adopted
            }
            agents.append(agent)
            agent_id += 1


    if max_agents is not None and len(agents) > max_agents:
        agents = agents[:max_agents]

    return agents










