import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D


def plot_adoption_rate(results_by_behavior):
    plt.figure(figsize=(8, 5))
    for name, results in results_by_behavior.items():
        plt.plot(results["adoption_rate"], label=name)
    plt.title("Overall Adoption Rate")
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative %")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/adoption_rate.png")
    plt.close()


def plot_new_adopters(results_by_behavior):
    plt.figure(figsize=(8, 5))
    for name, results in results_by_behavior.items():
        plt.plot(results["new_adopters"], label=name)
    plt.title("New Adopters per Step")
    plt.xlabel("Time Step")
    plt.ylabel("New Adopters")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("result/new_adopters.png")
    plt.close()

def plot_adoption_by_group(models_by_behavior):
    household_types = ["single", "couple_no_kids", "with_kids", "single_parent"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, ht in enumerate(household_types):
        ax = axs[i]
        for name, model in models_by_behavior.items():
            series = model.results["group_adoption"].get(ht, [])
            ax.plot(series, label=name)
        ax.set_title(f"Adoption Rate - {ht}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cumulative %")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("result/adoption_by_group.png")
    plt.close()


def plot_status_transitions(models_by_behavior):
    fig, axs = plt.subplots(1, len(models_by_behavior), figsize=(6 * len(models_by_behavior), 5))
    if len(models_by_behavior) == 1:
        axs = [axs]

    for ax, (name, model) in zip(axs, models_by_behavior.items()):
        for status, series in model.results["status_transitions"].items():
            time_series = [series.get(t, 0) for t in range(len(model.results["adoption_rate"]))]
            ax.plot(time_series, label=status)
        ax.set_title(f"Status Transition - {name}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Fraction of Agents")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("result/status_transitions.png")
    plt.close()

def plot_distributions(models_by_behavior):
    for name, model in models_by_behavior.items():
        dist = model.results["distributions"]
        for var in ["V", "S", "P"]:
            plt.figure(figsize=(8, 5))
            for t, values in enumerate(dist[var]):
                if t % 5 == 0:  # sample every 5 steps
                    plt.hist(values, bins=30, alpha=0.5, label=f"t={t}", density=True)
            plt.title(f"Distribution of {var} over Time ({name})")
            plt.xlabel(var)
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"result/dist_{var}_{name}.png")
            plt.close()

def plot_network(model, steps_to_plot=[0, 5, 10, 20, 30], save_dir="result"):
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
        plt.savefig(f"{save_dir}/network_t{t}_withdataandpolicy.png", dpi=300)

# def plot_beta_surface(beta_grid, final_adoptions, save_path="result/beta_surface.png"):
#     beta1_vals = sorted(set(b[0] for b in beta_grid))
#     beta2_vals = sorted(set(b[1] for b in beta_grid))
#     B1, B2 = np.meshgrid(beta1_vals, beta2_vals)
#     Z = np.array([[final_adoptions.get((b1, b2), 0) for b1 in beta1_vals] for b2 in beta2_vals])

#     fig = plt.figure(figsize=(10, 7))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(B1, B2, Z, cmap='viridis')
#     ax.set_xlabel("β₁")
#     ax.set_ylabel("β₂")
#     ax.set_zlabel("Adoption Rate")
#     ax.set_title("Adoption Surface by β Parameters")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

# def plot_beta(beta_grid, final_adoptions):
#     beta1_vals = sorted(set(b1 for b1, _ in beta_grid))
#     beta2_vals = sorted(set(b2 for _, b2 in beta_grid))

#     Z = np.array([[final_adoptions.get((b1, b2), 0)
#                    for b2 in beta2_vals] for b1 in beta1_vals])

#     B1, B2 = np.meshgrid(beta2_vals, beta1_vals)

#     plt.figure(figsize=(8, 6))
#     cp = plt.contourf(B2, B1, Z, cmap="viridis", levels=20)
#     plt.colorbar(cp, label='Final Adoption Rate')
#     plt.xlabel("Beta 2 (Social Influence Weight)")
#     plt.ylabel("Beta 1 (Economic Rationality Weight)")
#     plt.title("Adoption Rate across Beta Parameters")
#     plt.tight_layout()
#     plt.savefig("result/beta_contour.png", dpi=300)
#     plt.close()


def plot_beta(beta_grid, final_adoptions, save_path="result/beta_contour.png"):
    beta1_vals = sorted(set(b1 for b1, _ in beta_grid))
    beta2_vals = sorted(set(b2 for _, b2 in beta_grid))

    Z = np.array([[final_adoptions.get((b1, b2), 0)
                   for b2 in beta2_vals] for b1 in beta1_vals])

    B2, B1 = np.meshgrid(beta2_vals, beta1_vals)

    plt.figure(figsize=(9, 6))
    cp = plt.contourf(B2, B1, Z, levels=20, cmap='viridis')
    cs = plt.contour(B2, B1, Z, levels=10, colors='black', linewidths=0.5)
    plt.clabel(cs, fmt="%.2f", fontsize=8)

    max_pos = max(final_adoptions, key=final_adoptions.get)
    plt.plot(max_pos[1], max_pos[0], 'ro', markersize=6, label='Max')

    plt.colorbar(cp, label='Final Adoption Rate')
    plt.xlabel("Beta 2 (Social Influence Weight)", fontsize=12)
    plt.ylabel("Beta 1 (Economic Rationality Weight)", fontsize=12)
    plt.title("Adoption Rate across Beta Parameters", fontsize=14)
    plt.xticks(beta2_vals)
    plt.yticks(beta1_vals)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
