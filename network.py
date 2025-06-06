import random
from collections import defaultdict
from agent_simple import HouseholdAgent
import networkx as nx
import numpy as np

# group agents by location code
def group_by(agents, level="buurt"):
    group_map = defaultdict(list)
    for agent in agents:
        if level == "buurt":
            key = agent.location_code
        elif level == "wijk":
            key = agent.gemeente_code + agent.wijk_code
        elif level == "gemeente":
            key = agent.gemeente_code
        group_map[key].append(agent)
    return group_map


def assign_local(agents, level="buurt", max_local_neighbors=5):
    groups = group_by(agents, level)
    for group in groups.values():
        for agent in group:
            others = [a for a in group if a.id != agent.id]
            if max_local_neighbors is not None and len(others) > max_local_neighbors:
                neighbors = random.sample(others, k=max_local_neighbors)
            else:
                neighbors = others
            agent.neighbors = neighbors
            agent.n_neighbors = len(agent.neighbors)

def add_long_links(agents, k=5, filter_fn=None):
    id_map = {a.id: a for a in agents}
    ids = [a.id for a in agents]

    degree_weights = np.array([len(a.neighbors) for a in agents])
    degree_weights = degree_weights + 1 
    prob_dist = degree_weights / degree_weights.sum()

    for agent in agents:
        new_links = set()
        attempts = 0
        while len(new_links) < k and attempts < 50:
            candidate_id = np.random.choice(ids, p=prob_dist)
            candidate = id_map[candidate_id]
            attempts += 1

            if candidate.id == agent.id:
                continue
            if candidate in agent.neighbors:
                continue
            if candidate.location_code == agent.location_code:
                continue
            if filter_fn and not filter_fn(agent, candidate):
                continue

            agent.neighbors.append(candidate)
            new_links.add(candidate)

        agent.n_neighbors = len(agent.neighbors)



def build_net(agents, level="buurt", k=5, filter_fn=None, max_local_neighbors=5, return_nx=False):
    assign_local(agents, level=level, max_local_neighbors=max_local_neighbors)
    add_long_links(agents, k=k, filter_fn=filter_fn)


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
