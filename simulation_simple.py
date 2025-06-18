from model_simple import ABM
from config import n_agents, n_steps, beta, k_small_world, net_level
from visualization import plot_all_results, visualize_network_diffusion #, plot_network_graph, plot_spatial_heatmap

def run_simulation(behavior_mode):
    model = ABM(
        n_agents=n_agents,
        beta=beta,
        #policy_dict=policy_dict,
        #feed_mode="net_metering", 
        behavior_mode=behavior_mode,
        k_small_world=k_small_world,
        net_level=net_level
    )
    model.run(n_steps=n_steps)
    return model

if __name__ == "__main__":
    policy_list = [
        "no_policy",    
        "universal_nudge",
        "behavioral_push"
    ]


    all_results = {}
    models_by_strategy = {}

    for behavior_mode in policy_list:
        print(f"Running simulation for: {behavior_mode}")
        model = run_simulation(behavior_mode)
        models_by_strategy[behavior_mode] = model
        all_results[behavior_mode] = model.get_results()

    # for policy_dict in policy_list:
    #     print(f"Running simulation for: {policy_dict}")
    #     model = run_simulation(policy_dict)
    #     models_by_strategy[policy_dict] = model
    #     all_results[policy_dict] = model.get_results()

    plot_all_results(all_results)

    # focus_strategy = "support_vulnerable"
    #plot_network_graph(models_by_strategy[focus_strategy])
    #plot_spatial_heatmap(models_by_strategy[focus_strategy])
    visualize_network_diffusion(model, steps_to_plot=[0, 5, 10, 20, 30])


