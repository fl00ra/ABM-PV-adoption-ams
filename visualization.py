import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_all_results(strategy_results_dict):
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
    plt.show()


# def plot_network_graph(model, layout_seed=42):
#     G = model.nx_graph
#     pos = nx.spring_layout(G, seed=layout_seed)

#     adopted = {agent.id: agent.adopted for agent in model.agents}
#     colors = ["tab:orange" if adopted[n] else "tab:blue" for n in G.nodes]

#     plt.figure(figsize=(8, 6))
#     nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
#     nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=30)
#     plt.title("Network Structure (orange = adopted)")
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()


# def plot_spatial_heatmap(model):
#     data = {}
#     for agent in model.agents:
#         key = (agent.gemeente_code, agent.wijk_code)
#         if key not in data:
#             data[key] = [0, 0]
#         data[key][1] += 1
#         if agent.adopted:
#             data[key][0] += 1

#     municipalities = sorted({k[0] for k in data})
#     wijks = sorted({k[1] for k in data})

#     matrix = np.zeros((len(municipalities), len(wijks)))
#     for (m, w), (adopted, total) in data.items():
#         i = municipalities.index(m)
#         j = wijks.index(w)
#         matrix[i, j] = adopted / total if total > 0 else 0

#     plt.figure(figsize=(6, 4))
#     im = plt.imshow(matrix, cmap="OrRd", vmin=0, vmax=1)
#     plt.xticks(range(len(wijks)), wijks)
#     plt.yticks(range(len(municipalities)), municipalities)
#     plt.xlabel("Wijk Code")
#     plt.ylabel("Gemeente Code")
#     plt.title("Adoption Rate Heatmap")
#     plt.colorbar(im, label="Adoption Rate")
#     plt.tight_layout()
#     plt.show()
