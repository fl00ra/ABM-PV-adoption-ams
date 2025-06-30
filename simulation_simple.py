from model_simple import ABM
from config import n_agents, n_steps, k_small_world, net_level, beta
from visualization import (plot_adoption_by_group, plot_network, plot_adoption_rate, 
                          plot_new_adopters, plot_status_transitions, plot_distributions, 
                          plot_degree_distribution, compare_structure_behavior)


def run_simulation(behavior_mode, enable_feed_change=True):
    model = ABM(
        n_agents=n_agents,
        beta=beta,
        behavior_mode=behavior_mode,
        k_small_world=k_small_world,
        net_level=net_level,
        beta0_dist_type="unimodal",
        enable_feed_change=enable_feed_change,
    )
    model.run(n_steps=n_steps)
    return model


def run_dynamic_policy_simulation():
    model = ABM(
        n_agents=n_agents,
        beta=beta,
        behavior_mode="universal_nudge",  
        k_small_world=k_small_world,
        net_level=net_level,
        beta0_dist_type="unimodal",
        enable_feed_change=True,
    )
    
    for t in range(n_steps):
        if t == 10:
            model.policy.behavior_mode = "energy_subsidies"
            print(f"\n[T={t}] Switched to energy_subsidies policy")
        elif t == 20:
            model.policy.behavior_mode = "progressive_subsidy"
            print(f"\n[T={t}] Switched to progressive_subsidy policy")
        
        model.step(t)
        model._record(t)
    
    return model

if __name__ == "__main__":
    policy_list = [
        "no_policy",    
        "universal_nudge",
        "energy_subsidies",
        # "progressive_subsidy",
        # "dynamic_subsidy",
        "time_limited_subsidy"
    ]


    results_by_behavior = {}
    models_by_behavior = {}

    for behavior_mode in policy_list:
        print(f"Running simulation for: {behavior_mode}")
        model = run_simulation(behavior_mode)
        models_by_behavior[behavior_mode] = model
        results_by_behavior[behavior_mode] = model.get_results()

        plot_network(
            model,
            steps_to_plot=[0, 5, 10, 20, 30],
            save_dir=f"result/nxincome/network_{behavior_mode}",
            label=behavior_mode
        )


    plot_adoption_rate(results_by_behavior)
    plot_new_adopters(results_by_behavior)
    plot_adoption_by_group(models_by_behavior)
    plot_status_transitions(models_by_behavior)
    plot_distributions(models_by_behavior)   
    # plot_network(model, steps_to_plot=[0, 5, 10, 20, 30])
    # plot_degree_distribution(model.agents)
    plot_degree_distribution(models_by_behavior[policy_list[0]].agents)


    model_uni = ABM(
        n_agents=n_agents,
        beta=beta,
        behavior_mode="no_policy",  
        k_small_world=k_small_world,
        net_level=net_level,
        beta0_dist_type="unimodal"
        )
    model_uni.run(n_steps=30)

    model_bi = ABM(
        n_agents=n_agents,
        beta=beta,
        behavior_mode="no_policy",  
        k_small_world=k_small_world,
        net_level=net_level,
        beta0_dist_type="bimodal"
        )
    model_bi.run(n_steps=30)


    models_by_structure = {
        "Unimodal": model_uni,
        "Bimodal": model_bi
        }

    compare_structure_behavior(models_by_structure)