import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from network_geo import assign_local, add_long_links
import geopandas as gpd
import matplotlib.animation as animation
import os
from networkx.drawing.nx_agraph import graphviz_layout
import random
import pandas as pd
import seaborn as sns




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
    plt.savefig("result/npv/adoption_rate.png")
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
    plt.savefig("result/npv/new_adopters.png")
    plt.close()


def plot_adoption_by_group(models_by_behavior):
    household_types = ["single", "couple_no_kids", "with_kids", "single_parent"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, ht in enumerate(household_types):
        ax = axs[i]
        for name, model in models_by_behavior.items():
            series = model.results["group_adoption"].get(ht, [])

            series = np.array(series)  # try convert to array
            if series.ndim == 1:
                # fallback to single run
                ax.plot(series, label=name)
            elif series.ndim == 2:
                # multiple runs → mean + CI
                mean = np.mean(series, axis=0)
                std_err = np.std(series, axis=0, ddof=1) / np.sqrt(series.shape[0])
                ci_upper = mean + 1.96 * std_err
                ci_lower = mean - 1.96 * std_err

                ax.plot(mean, label=name)
                ax.fill_between(np.arange(len(mean)), ci_lower, ci_upper, alpha=0.25)
            else:
                print(f"[Warning] Unexpected shape for {ht} - {name}: {series.shape}")

        ax.set_title(f"Adoption Rate - {ht}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Cumulative %")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("result/npv/adoption_by_group_with_ci.png")
    plt.close()




# def plot_adoption_by_group(models_by_behavior):
#     household_types = ["single", "couple_no_kids", "with_kids", "single_parent"]
#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     axs = axs.flatten()

#     for i, ht in enumerate(household_types):
#         ax = axs[i]
#         for name, model in models_by_behavior.items():
#             series = model.results["group_adoption"].get(ht, [])
#             ax.plot(series, label=name)
#         ax.set_title(f"Adoption Rate - {ht}")
#         ax.set_xlabel("Time Step")
#         ax.set_ylabel("Cumulative %")
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.savefig("result/trial/adoption_by_group.png")
#     plt.close()


# def plot_status_transitions(models_by_behavior):
#     fig, axs = plt.subplots(1, len(models_by_behavior), figsize=(6 * len(models_by_behavior), 5))
#     if len(models_by_behavior) == 1:
#         axs = [axs]

#     for ax, (name, model) in zip(axs, models_by_behavior.items()):
#         for status, series in model.results["status_transitions"].items():
#             time_series = [series.get(t, 0) for t in range(len(model.results["adoption_rate"]))]
#             ax.plot(time_series, label=status)
#         ax.set_title(f"Status Transition - {name}")
#         ax.set_xlabel("Time Step")
#         ax.set_ylabel("Fraction of Agents")
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.savefig("result/nxincome/status_transitions.png")
#     plt.close()

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
            plt.savefig(f"result/npv/dist_{var}_{name}.png")
            plt.close()

def plot_network(model, steps_to_plot=[0, 5, 10, 20, 30], save_dir="result/npv", label=""):
    """
    Plots the network at specific time steps with color showing adoption.
    """
    os.makedirs(save_dir, exist_ok=True)

    G = model.nx_graph
    pos = nx.spring_layout(G, seed=42)  

    for t in steps_to_plot:
        plt.figure(figsize=(10, 8))
        adopted_nodes = {a.id for a in model.agents if a.adoption_time is not None and a.adoption_time <= t}
        colors = ["red" if node in adopted_nodes else "lightgray" for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=20, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.3)
        plt.title(f"Network Diffusion at Time Step {t} ({label})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/network_t{t}.png", dpi=300)



def plot_beta(beta_grid, final_adoptions, save_path="result/npv/beta_contour.png"):
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

def plot_degree_distribution(agents, save_path="result/npv/degree_distribution.png"):
    """
    Plots the histogram of node degrees in the social network.
    """
    degrees = [len(agent.neighbors) for agent in agents]
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=range(min(degrees), max(degrees)+2), density=True,
             edgecolor='black', alpha=0.75)
    plt.title("Degree Distribution of the Social Network")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def export_agents_for_mapping(model, filepath):
    """
    Export agent data for geospatial visualization.
    """
    data = []
    for agent in model.agents:
        row = {
            "id": agent.id,
            "postcode6": agent.postcode6,
            "buurt_code": agent.buurt_code,
            "adoption_time": agent.adoption_time,
            "social_weight": agent.social_weight,
            "household_type": agent.household_type,
            "lihe": agent.lihe,
        }
        for t, adopted in enumerate(agent.adoption_track):
            row[f"adoption_t{t}"] = int(adopted)
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Exported agent data to: {filepath}")


# def plot_static_adoption_map(agent_csv_path, geo_path, timestep, output_path=None):
#     """
#     绘制某一时间步下的 PV 采纳比例地图（基于 buurt_code 匹配 CBS shapefile）
#     """
#     agents = pd.read_csv(agent_csv_path)
#     gdf = gpd.read_file(geo_path)

#     # 从 cbs_code 提取后缀（如 BU036300 → 036300）
#     gdf["buurt_code"] = gdf["cbs_code"].str[2:]

#     # 聚合采纳数据
#     agents_grouped = agents.groupby("buurt_code")[f"adoption_t{timestep}"].mean().reset_index()
#     agents_grouped["buurt_code"] = agents_grouped["buurt_code"].astype(str)

#     merged = gdf.merge(agents_grouped, on="buurt_code", how="left")

#     # 绘图
#     fig, ax = plt.subplots(figsize=(10, 10))
#     merged.plot(column=f"adoption_t{timestep}",
#                 cmap="YlGn",
#                 linewidth=0.2,
#                 edgecolor='grey',
#                 legend=True,
#                 ax=ax)

#     ax.set_title(f"PV Adoption Rate by Buurt (Timestep {timestep})", fontsize=14)
#     ax.axis("off")

#     if output_path:
#         plt.savefig(output_path, dpi=300)
#         print(f"Saved to {output_path}")
#     else:
#         plt.show()

# def plot_beta0_distribution(beta0_values, title="Distribution of β₀"):
#     plt.figure(figsize=(6, 4))
#     sns.histplot(beta0_values, bins=30, kde=True)
#     plt.xlabel("β₀ (Baseline Preference)")
#     plt.ylabel("Frequency")
#     plt.title(title)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("result/beta0_dist_unimodal.png")
#     plt.show()
    

def compare_structure_behavior(models_by_structure, save_path="result/npv/structure_behavior_comparison.png"):
    """
    Compare beta₀ distributions and corresponding adoption curves in one figure.
    Each column = one preference structure (e.g., unimodal vs bimodal)
    """
    n = len(models_by_structure)
    fig, axs = plt.subplots(2, n, figsize=(5 * n, 8))
    
    for i, (label, model) in enumerate(models_by_structure.items()):
        sns.histplot(model.beta0_values, bins=30, kde=True, ax=axs[0, i])
        axs[0, i].set_title(f"β₀ Distribution: {label}")
        axs[0, i].set_xlabel("β₀")
        axs[0, i].set_ylabel("Frequency")
        axs[0, i].grid(True)

        axs[1, i].plot(model.results["adoption_rate"], label="Adoption Rate")
        axs[1, i].set_title(f"Adoption Over Time: {label}")
        axs[1, i].set_xlabel("Time Step")
        axs[1, i].set_ylabel("Cumulative %")
        axs[1, i].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison figure to: {save_path}")

def plot_policy_impact_over_time(model, save_path="result/nxincome/policy_impact.png"):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    sample_agents = random.sample(model.agents, min(5, len(model.agents)))
    
    for agent in sample_agents:
        costs = []
        for t in range(len(model.results["adoption_rate"])):
            effective_cost = model.policy.get_effective_cost(agent, t)
            costs.append(effective_cost)
        ax1.plot(costs, label=f"Agent {agent.id[:8]}")
    
    ax1.set_title("Effective Cost Over Time (Sample Agents)")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Effective Cost")
    ax1.legend()
    ax1.grid(True)
    
    avg_costs = []
    for t in range(len(model.results["adoption_rate"])):
        costs_at_t = [model.policy.get_effective_cost(a, t) for a in model.agents]
        avg_costs.append(np.mean(costs_at_t))
    
    ax2.plot(avg_costs, 'b-', linewidth=2)
    ax2.fill_between(range(len(avg_costs)), 
                     [np.percentile([model.policy.get_effective_cost(a, t) for a in model.agents], 25) 
                      for t in range(len(avg_costs))],
                     [np.percentile([model.policy.get_effective_cost(a, t) for a in model.agents], 75) 
                      for t in range(len(avg_costs))],
                     alpha=0.3)
    ax2.set_title("Average Effective Cost Over Time")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Average Effective Cost")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()