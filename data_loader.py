import pandas as pd
import numpy as np
import random
from config import n_agents


path = "Processed_ams_pc6_buurt_data.csv"

def sample_income():
    # log(30000) ≈ 10.3
    mu = 10.3
    sigma = 0.45
    return np.random.lognormal(mean=mu, sigma=sigma)

# def sample_income():
#     p = random.random()
#     if p < 0.42:
#         return np.random.normal(30000, 5000)  # low 
#     elif p < 0.58:
#         return np.random.normal(45000, 4000)  # middle-low
#     elif p < 0.71:
#         return np.random.normal(55000, 4000)  # middle-high
#     else:
#         return np.random.normal(75000, 10000)  # high
    

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
            income = max(10000, sample_income())
            energielabel = random.choices(list("ABCDEFG"), weights=[0.15, 0.2, 0.25, 0.2, 0.1, 0.07, 0.03])[0]
            adopted = random.random() < 0.25

            agent = {
                "id": agent_id,
                "income": income,
                "energielabel": energielabel,
                "elek_usage": elek_usage_hh[i],
                "elec_price": 0.4,
                "household_type": all_types[i % len(all_types)],
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












# import pandas as pd
# import numpy as np

# def load_amsterdam_data(path="Processed_ams_pc6_buurt_data.csv", n_agents=5000):
#     df = pd.read_csv(path)
#     df = df.dropna(subset=["buurt_code"])

#     # Normalize postcode format
#     df["postcode6"] = df["postcode6"].astype(str).str.replace(r"\\s+", "", regex=True).str.upper()

#     # Get all buurt codes
#     buurt_groups = df.groupby("buurt_code")

#     agents = []
#     agent_id = 0

#     for buurt, group in buurt_groups:
#         total_hh = group["total_households"].sum()
#         if total_hh == 0:
#             continue

#         # Compute proportions
#         probs = {
#             "single": group["single"].sum() / total_hh,
#             "couple_no_kids": group["couple_no_kids"].sum() / total_hh,
#             "with_kids": group["with_kids"].sum() / total_hh,
#             "single_parent": group["single_parent"].sum() / total_hh
#         }
#         probs = {k: max(v, 0) for k, v in probs.items()}
#         norm_factor = sum(probs.values())
#         probs = {k: v / norm_factor for k, v in probs.items()}

#         vulnerable_ratio = group["vulnerable"].sum() / total_hh if "vulnerable" in group else 0.0

#         n_buurt_agents = min(int(total_hh), n_agents - len(agents))
#         if n_buurt_agents <= 0:
#             break

#         household_types = np.random.choice(list(probs.keys()), size=n_buurt_agents, p=list(probs.values()))
#         vulnerable_flags = np.random.rand(n_buurt_agents) < vulnerable_ratio


#         elek_total = group["elek_usage"].sum()
#         if elek_total <= 0:
#             elek_distribution = np.random.normal(2500, 500, size=n_buurt_agents)
#         else:
#             elek_distribution = np.random.dirichlet(np.ones(n_buurt_agents)) * elek_total

#         # elek_distribution = np.random.dirichlet(np.ones(n_buurt_agents)) * elek_total

#         for i in range(n_buurt_agents):
#             agents.append({
#                 "id": agent_id,
#                 "income": np.random.normal(35000, 12000),
#                 "energielabel": np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.05, 0.05]),
#                 "elek_usage": elek_distribution[i],
#                 "elec_price": 0.4,
#                 "household_type": household_types[i],
#                 "lihe": int(vulnerable_flags[i]),
#                 "postcode6": group["postcode6"].iloc[i % len(group)],
#                 "buurt_code": buurt,
#                 "adopted": np.random.rand() < 0.25
#             })
#             agent_id += 1

#             if len(agents) >= n_agents:
#                 break

#     return agents







