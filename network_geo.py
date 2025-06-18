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
    return max(1, np.random.poisson(lam=6))

def assign_local(agents, level="buurt", local_ratio=0.85):
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
                neighbors = random.sample(others, k=k_local)
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
            if candidate in agent.neighbors:
                continue
            if candidate.buurt_code == agent.buurt_code:
                continue
            if filter_fn and not filter_fn(agent, candidate):
                continue

            agent.neighbors.append(candidate)
            new_links.add(candidate)

        agent.n_neighbors = len(agent.neighbors)

def build_net(agents, level="buurt", filter_fn=None, return_nx=False):
    agent_k_map = assign_local(agents, level=level)
    add_long_links(agents, agent_k_map, filter_fn=filter_fn)

    if return_nx:
        G = nx.Graph()
        G.add_nodes_from([a.id for a in agents])

        added_edges = set()
        for agent in agents:
            for neighbor in agent.neighbors:
                edge = tuple(sorted((agent.id, neighbor.id)))
                if edge not in added_edges:
                    G.add_edge(*edge)
                    added_edges.add(edge)

        return G