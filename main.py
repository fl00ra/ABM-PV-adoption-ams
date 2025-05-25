import matplotlib.pyplot as plt
import numpy as np
from model import ABM
from config import n_agents, n_steps, beta, gamma, k_small_world, net_level

def run_simulation(strategy_tag):
    model = ABM(
        n_agents=n_agents,
        beta=beta,
        gamma=gamma,
        policy_dict={"strategy_tag": strategy_tag},
        k_small_world=k_small_world,
        net_level=net_level
    )
    model.run(n_steps=n_steps)
    results = model.get_results()
    return results

def plot_all_results(strategy_results_dict):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for strategy, result in strategy_results_dict.items():
        label = strategy.replace("_", " ").title()
        axs[0].plot(result["adoption_rate"], label=label)
        axs[1].bar(np.arange(len(result["new_adopters"])), result["new_adopters"], alpha=0.4, label=label)
        axs[2].plot(result["visible_rate"], label=label)
        axs[3].plot(result["targeted_adoption_rate"], label=label)

    axs[0].set_title("Overall Adoption Rate")
    axs[0].set_ylabel("Cumulative %")
    axs[0].grid(True)

    axs[1].set_title("New Adopters Per Step")
    axs[1].set_ylabel("New Adopters")
    axs[1].grid(True)

    axs[2].set_title("Visible Agent Ratio")
    axs[2].set_ylabel("Proportion visible")
    axs[2].grid(True)

    axs[3].set_title("Targeted Adoption Rate")
    axs[3].set_ylabel("Targeted %")
    axs[3].grid(True)

    for ax in axs:
        ax.set_xlabel("Time Step")
        ax.legend()
        ax.set_xlim(left=0)

    plt.suptitle("Solar PV Adoption Dynamics by Strategy", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    strategy_list = [
        "no_policy",
        "fast_adoption",
        "equity_first",
        "universal_nudge",
        "behavioral_first"
    ]

    all_results = {}

    for strategy in strategy_list:
        result = run_simulation(strategy_tag=strategy)
        all_results[strategy] = result

    plot_all_results(all_results)
