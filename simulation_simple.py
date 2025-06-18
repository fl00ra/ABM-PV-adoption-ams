from model_simple import ABM
from config import n_agents, n_steps, beta, k_small_world, net_level
from visualization import plot_all_results, visualize_network_diffusion #, plot_network_graph, plot_spatial_heatmap

def run_simulation(strategy_tag):
    model = ABM(
        n_agents=n_agents,
        beta=beta,
        policy_dict={"strategy_tag": strategy_tag},
        k_small_world=k_small_world,
        net_level=net_level
    )
    model.run(n_steps=n_steps)
    return model

if __name__ == "__main__":
    strategy_list = [
        "no_policy",
        "reduce_cost"
        ]

    all_results = {}
    models_by_strategy = {}

    for strategy in strategy_list:
        print(f"Running simulation for: {strategy}")
        model = run_simulation(strategy_tag=strategy)
        models_by_strategy[strategy] = model
        all_results[strategy] = model.get_results()

    plot_all_results(all_results)

    # focus_strategy = "support_vulnerable"
    #plot_network_graph(models_by_strategy[focus_strategy])
    #plot_spatial_heatmap(models_by_strategy[focus_strategy])
    visualize_network_diffusion(model, steps_to_plot=[0, 5, 10, 20, 30])
