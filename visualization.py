import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

def plot_all_results(strategy_results_dict, save_path="result/adoption_3status.png"):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs = axs.flatten()

    for strategy, result in strategy_results_dict.items():
        label = strategy.replace("_", " ").title()
        axs[0].plot(result["adoption_rate"], label=label)
        axs[1].plot(result["new_adopters"], label=label)
        axs[2].plot(result["targeted_adoption_rate"], label=label)

    axs[0].set_title("Overall Adoption Rate")
    axs[0].set_ylabel("Cumulative %")
    axs[0].set_xlabel("Time Step")
    axs[0].grid(True)

    axs[1].set_title("New Adopters Per Step")
    axs[1].set_ylabel("New Adopters")
    axs[1].set_xlabel("Time Step")
    axs[1].grid(True)

    axs[2].set_title("Targeted Adoption Rate")
    axs[2].set_ylabel("Targeted %")
    axs[2].set_xlabel("Time Step")
    axs[2].grid(True)

    for ax in axs:
        ax.legend()
        ax.set_xlim(left=0)

    plt.suptitle("Solar PV Adoption Dynamics by Strategy", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def visualize_network_diffusion(model, steps_to_plot=[0, 5, 10, 20, 30], save_dir="result"):
    """
    Plots the network at specific time steps with color showing adoption.
    """
    G = model.nx_graph
    pos = nx.spring_layout(G, seed=42)  

    for t in steps_to_plot:
        plt.figure(figsize=(10, 8))
        adopted_nodes = {a.id for a in model.agents if a.adoption_time is not None and a.adoption_time <= t}
        colors = ["red" if node in adopted_nodes else "lightgray" for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=20, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.3)
        plt.title(f"Network Diffusion at Time Step {t}", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/network_t{t}_3status.png", dpi=300)
        # plt.show()
