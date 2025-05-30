n_agents = 200
n_steps = 30

beta = [-2.5, 1.2, 2.5, 0.2]  # β₀, β₁, β₂, β₃
gamma = [0.6, -0.4, -0.3, -0.5, 0.3, 0.5, 0.2]
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