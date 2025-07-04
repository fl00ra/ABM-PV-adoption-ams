import pandas as pd
import numpy as np

# path = "Processed_ams_pc6_buurt_data.csv"

# def compute_n_agents():
#     df = pd.read_csv(path).drop_duplicates(subset=["postcode6", "buurt_code"])
#     return int(df["total_households"].sum())

# n_agents = compute_n_agents()

n_agents = 5000
n_steps = 30


# beta1_range = [0.2, 0.6, 1.0, 1.4, 1.8, 2.2]
# beta2_range = [0.7, 1.4, 2.1, 2.8, 2., 3.0]

beta1_range = np.linspace(0.0002, 0.0015, num=9) 
beta2_range = np.linspace(2.0, 6.0, num=10) 

beta = [0.0002, 5]  # β₁, β₂

# THETA = 1000


k_small_world = 2
net_level = "buurt"  # or "wijk", "gemeente"

# household_type_map = {
#     "single": [0, 0, 0],
#     "couple_no_kids": [1, 0, 0],
#     "with_kids": [0, 1, 0],
#     "nonfamily_group": [0, 0, 1]
# }