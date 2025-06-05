n_agents = 5000
n_steps = 50

beta = [0.8, 2.0]  # β₁, β₂
# gamma = [0.6, -0.4, -0.3, -0.5, 0.3, 0.5, 0.2]
# gamma = [income, lihe, lekwi, lihezlek, no_kids, with_kids, nonfamily_group]
THETA = 1000


k_small_world = 2
net_level = "buurt"  # or "wijk", "gemeente"

household_type_map = {
    "single": [0, 0, 0],
    "couple_no_kids": [1, 0, 0],
    "with_kids": [0, 1, 0],
    "nonfamily_group": [0, 0, 1]
}