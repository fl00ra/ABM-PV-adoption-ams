from model_simple import ABM
from config import n_agents, n_steps, k_small_world, net_level, beta1_range, beta2_range
from visualization import plot_beta

beta_grid = [(b1, b2) for b1 in beta1_range for b2 in beta2_range]
final_adoptions = {}

for b1, b2 in beta_grid:
    print(f"Running for beta1={b1}, beta2={b2}")
    model = ABM(
        n_agents=n_agents,
        beta=[b1, b2],
        behavior_mode="no_policy",
        k_small_world=k_small_world,
        net_level=net_level
    )
    model.run(n_steps=n_steps)
    final_adoptions[(b1, b2)] = model.get_results()["adoption_rate"][-1]

plot_beta(beta_grid, final_adoptions)

