# import random
# from collections import defaultdict
# import networkx as nx
# import numpy as np

# def group_by(agents, level="buurt"):
#     group_map = defaultdict(list)
#     for agent in agents:
#         if level == "buurt":
#             key = agent.buurt_code
#         elif level == "wijk":
#             key = agent.buurt_code[:6]
#         elif level == "gemeente":
#             key = agent.buurt_code[:4]
#         else:
#             key = agent.buurt_code
#         group_map[key].append(agent)
#     return group_map


# # def total_deg(alpha=2.5, min_k=2, max_k=100):
# #     # Draw from Pareto(alpha), then scale
# #     raw = (np.random.pareto(alpha) + 1) * min_k
# #     degree = int(np.clip(raw, min_k, max_k))  # Limit max degree to avoid instability
# #     return degree

# def total_deg():
#     degree = int(round(np.random.lognormal(mean=np.log(7), sigma=0.5)))
#     return max(2, degree)


# # def assign_local(agents, level="buurt", local_ratio=0.7):
# #     groups = group_by(agents, level)
# #     agent_k_map = {}
# #     for agent in agents:
# #         k_total = total_deg()
# #         k_local = np.random.binomial(k_total, p=local_ratio)
# #         k_long = k_total - k_local
# #         agent_k_map[agent.id] = (k_local, k_long)

# #     for group in groups.values():
# #         group_ids = set(a.id for a in group)
# #         for agent in group:
# #             others = [a for a in group if a.id != agent.id]
# #             k_local, _ = agent_k_map[agent.id]
# #             if len(others) > k_local:
# #                 neighbors = random.sample(others, k=k_local)
# #             else:
# #                 neighbors = others
# #             agent.neighbors = neighbors
# #             agent.n_neighbors = len(agent.neighbors)

# #     return agent_k_map

# def assign_local(agents, level="buurt", local_ratio=0.7):
#     def similarity(agent1, agent2):
#         """Calculate similarity between two agents based on household type."""
#         return 1.0 if agent1.household_type == agent2.household_type else 0.3

#     groups = group_by(agents, level)
#     agent_k_map = {}

#     for agent in agents:
#         k_total = total_deg()
#         k_local = np.random.binomial(k_total, p=local_ratio)
#         k_long = k_total - k_local
#         agent_k_map[agent.id] = (k_local, k_long)

#     for group in groups.values():
#         group_ids = set(a.id for a in group)
#         for agent in group:
#             others = [a for a in group if a.id != agent.id]
#             k_local, _ = agent_k_map[agent.id]

#             if len(others) > k_local:
#                 weights = np.array([similarity(agent, other) for other in others])
#                 weights = weights / weights.sum()
#                 neighbors = list(np.random.choice(others, size=k_local, replace=False, p=weights))
#             else:
#                 neighbors = others

#             agent.neighbors = neighbors
#             agent.n_neighbors = len(agent.neighbors)

#     return agent_k_map


# # def add_long_links(agents, agent_k_map, filter_fn=None):
# #     id_map = {a.id: a for a in agents}
# #     ids = [a.id for a in agents]

# #     degree_weights = np.array([len(a.neighbors) for a in agents]) + 1
# #     prob_dist = degree_weights / degree_weights.sum()

# #     for agent in agents:
# #         k_local, k_long = agent_k_map[agent.id]
# #         new_links = set()
# #         attempts = 0
# #         while len(new_links) < k_long and attempts < 50:
# #             candidate_id = np.random.choice(ids, p=prob_dist)
# #             candidate = id_map[candidate_id]
# #             attempts += 1

# #             if candidate.id == agent.id:
# #                 continue
# #             if candidate in agent.neighbors:
# #                 continue
# #             if candidate.buurt_code == agent.buurt_code:
# #                 continue
# #             if filter_fn and not filter_fn(agent, candidate):
# #                 continue

# #             agent.neighbors.append(candidate)
# #             candidate.neighbors.append(agent)

# #             new_links.add(candidate)

# #         agent.n_neighbors = len(agent.neighbors)
        
# def add_long_links(agents, agent_k_map, filter_fn=None):
#     id_map = {a.id: a for a in agents}
#     ids = [a.id for a in agents]

#     degree_weights = np.array([len(a.neighbors) for a in agents]) + 1
#     prob_dist = degree_weights / degree_weights.sum()

#     existing_links = {a.id: set(n.id for n in a.neighbors) for a in agents}

#     for agent in agents:
#         k_local, k_long = agent_k_map[agent.id]
#         new_links = set()
#         attempts = 0

#         while len(new_links) < k_long and attempts < 50:
#             candidate_id = np.random.choice(ids, p=prob_dist)
#             candidate = id_map[candidate_id]
#             attempts += 1

#             if candidate.id == agent.id:
#                 continue
#             if candidate.id in existing_links[agent.id]:
#                 continue
#             if candidate.buurt_code == agent.buurt_code:
#                 continue
#             if filter_fn and not filter_fn(agent, candidate):
#                 continue

#             agent.neighbors.append(candidate)
#             candidate.neighbors.append(agent)

#             existing_links[agent.id].add(candidate.id)
#             existing_links[candidate.id].add(agent.id)

#             new_links.add(candidate)

#         agent.n_neighbors = len(agent.neighbors)

# # def build_net(agents, level="buurt", filter_fn=None, return_nx=False):
# #     agent_k_map = assign_local(agents, level=level)
# #     add_long_links(agents, agent_k_map, filter_fn=filter_fn)

# #     if return_nx:
# #         G = nx.Graph()
# #         G.add_nodes_from([a.id for a in agents])

# #         added_edges = set()
# #         for agent in agents:
# #             for neighbor in agent.neighbors:
# #                 edge = tuple(sorted((agent.id, neighbor.id)))
# #                 if edge not in added_edges:
# #                     G.add_edge(*edge)
# #                     added_edges.add(edge)


# #         return G
        
# def build_net(agents, level="buurt", filter_fn=None, return_nx=False, assign_social_weight=True):
#     agent_k_map = assign_local(agents, level=level)
#     add_long_links(agents, agent_k_map, filter_fn=filter_fn)

#     if assign_social_weight or return_nx:
#         G = nx.Graph()
#         G.add_nodes_from([a.id for a in agents])

#         added_edges = set()
#         for agent in agents:
#             for neighbor in agent.neighbors:
#                 edge = tuple(sorted((agent.id, neighbor.id)))
#                 if edge not in added_edges:
#                     G.add_edge(*edge)
#                     added_edges.add(edge)

#         if assign_social_weight:
#             centrality = nx.eigenvector_centrality(G, max_iter=1000)
#             agent_map = {a.id: a for a in agents}
#             for agent_id, value in centrality.items():
#                 agent_map[agent_id].social_weight = value

#         if return_nx:
#             return G

import random
from collections import defaultdict
import networkx as nx
import numpy as np

def group_by(agents, level="buurt"):
    group_map = defaultdict(list)
    for agent in agents:
        if level == "buurt":
            key = agent.buurt_code
        elif level == "wijk":
            key = agent.buurt_code[:6]
        elif level == "gemeente":
            key = agent.buurt_code[:4]
        else:
            key = agent.buurt_code
        group_map[key].append(agent)
    return group_map

def total_deg():
    degree = int(round(np.random.lognormal(mean=np.log(7), sigma=0.5)))
    return max(2, degree)

# def assign_local(agents, level="buurt", local_ratio=0.7, use_income=False):
#     def similarity(agent1, agent2):
#         household_score = 1.0 if agent1.household_type == agent2.household_type else 0.3
#         income_score = 1.0 if agent1.income_level == agent2.income_level else 0.3
#         if use_income:
#             return 0.4 * household_score + 0.6 * income_score
#         else:
#             return household_score


def assign_local(agents, level="buurt", local_ratio=0.7, hh_weight=0.1, income_weight=0.9):
    def similarity(agent1, agent2):
        household_score = 1.0 if agent1.household_type == agent2.household_type else 0.3
        income_score = 1.0 if agent1.income_level == agent2.income_level else 0.3
        return hh_weight * household_score + income_weight * income_score

    groups = group_by(agents, level)
    agent_k_map = {}

    for agent in agents:
        k_total = total_deg()
        k_local = np.random.binomial(k_total, p=local_ratio)
        k_long = k_total - k_local
        agent_k_map[agent.id] = (k_local, k_long)

    for group in groups.values():
        group_ids = set(a.id for a in group)
        for agent in group:
            others = [a for a in group if a.id != agent.id]
            k_local, _ = agent_k_map[agent.id]

            if len(others) > k_local:
                weights = np.array([similarity(agent, other) for other in others])
                weights = weights / weights.sum()
                neighbors = list(np.random.choice(others, size=k_local, replace=False, p=weights))
            else:
                neighbors = others

            agent.neighbors = neighbors
            agent.n_neighbors = len(agent.neighbors)

    return agent_k_map

def add_long_links(agents, agent_k_map, filter_fn=None):
    id_map = {a.id: a for a in agents}
    ids = [a.id for a in agents]

    degree_weights = np.array([len(a.neighbors) for a in agents]) + 1
    prob_dist = degree_weights / degree_weights.sum()

    existing_links = {a.id: set(n.id for n in a.neighbors) for a in agents}

    for agent in agents:
        k_local, k_long = agent_k_map[agent.id]
        new_links = set()
        attempts = 0

        while len(new_links) < k_long and attempts < 50:
            candidate_id = np.random.choice(ids, p=prob_dist)
            candidate = id_map[candidate_id]
            attempts += 1

            if candidate.id == agent.id:
                continue
            if candidate.id in existing_links[agent.id]:
                continue
            if candidate.buurt_code == agent.buurt_code:
                continue
            if filter_fn and not filter_fn(agent, candidate):
                continue

            agent.neighbors.append(candidate)
            candidate.neighbors.append(agent)

            existing_links[agent.id].add(candidate.id)
            existing_links[candidate.id].add(agent.id)

            new_links.add(candidate)

        agent.n_neighbors = len(agent.neighbors)

def build_net(agents, level="buurt", filter_fn=None, return_nx=False, assign_social_weight=True, hh_weight=0.1, income_weight=0.9):
    agent_k_map = assign_local(agents, level=level, hh_weight=hh_weight, income_weight=income_weight)
    add_long_links(agents, agent_k_map, filter_fn=filter_fn)

    if assign_social_weight or return_nx:
        G = nx.Graph()
        G.add_nodes_from([a.id for a in agents])

        added_edges = set()
        for agent in agents:
            for neighbor in agent.neighbors:
                edge = tuple(sorted((agent.id, neighbor.id)))
                if edge not in added_edges:
                    G.add_edge(*edge)
                    added_edges.add(edge)

        if assign_social_weight:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
            agent_map = {a.id: a for a in agents}
            for agent_id, value in centrality.items():
                agent_map[agent_id].social_weight = value

        if return_nx:
            return G
