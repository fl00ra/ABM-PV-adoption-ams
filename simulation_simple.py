from model_simple import ABM
from config import n_agents, n_steps, k_small_world, net_level, beta
from visualization import plot_adoption_by_group, plot_network, plot_adoption_rate, plot_new_adopters, plot_status_transitions, plot_distributions

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


    results_by_behavior = {}
    models_by_behavior = {}

    for behavior_mode in policy_list:
        print(f"Running simulation for: {behavior_mode}")
        model = run_simulation(behavior_mode)
        models_by_behavior[behavior_mode] = model
        results_by_behavior[behavior_mode] = model.get_results()


    plot_adoption_rate(results_by_behavior)
    plot_new_adopters(results_by_behavior)
    plot_adoption_by_group(models_by_behavior)
    plot_status_transitions(models_by_behavior)
    plot_distributions(models_by_behavior)   
    plot_network(model, steps_to_plot=[0, 5, 10, 20, 30])
