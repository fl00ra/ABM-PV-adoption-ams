import pandas as pd

# path = "Processed_ams_pc6_buurt_data.csv"

# def compute_n_agents():
#     df = pd.read_csv(path).drop_duplicates(subset=["postcode6", "buurt_code"])
#     return int(df["total_households"].sum())

# n_agents = compute_n_agents()

n_agents = 1000
n_steps = 40


beta1_range = [0.2, 0.4, 0.6, 0.8, 1.0]
beta2_range = [0.5, 1.0, 1.5, 2.0]

beta = [0.2, 0.8]  # β₁, β₂

THETA = 2000


k_small_world = 2
net_level = "buurt"  # or "wijk", "gemeente"

# household_type_map = {
#     "single": [0, 0, 0],
#     "couple_no_kids": [1, 0, 0],
#     "with_kids": [0, 1, 0],
#     "nonfamily_group": [0, 0, 1]
# }