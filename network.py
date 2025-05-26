import random
from collections import defaultdict
from agent import HouseholdAgent
import networkx as nx

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

# allocate local connections within the group
def assign_local(agents, level="buurt"):
    groups = group_by(agents, level)
    for group in groups.values():
        for agent in group:
            agent.neighbors = [a for a in group if a.id != agent.id]
            agent.n_neighbors = len(agent.neighbors)

# add long-range connections 
def add_long_links(agents, k=4, filter_fn=None):
    id_map = {a.id: a for a in agents}
    ids = [a.id for a in agents]

    for agent in agents:
        new_links = set()
        while len(new_links) < k:
            candidate = id_map[random.choice(ids)]

            if candidate.id == agent.id:
                continue
            if candidate in agent.neighbors:
                continue

            # not connect to buurt
            if candidate.location_code == agent.location_code:
                continue
            
            # defined later in model.py
            if filter_fn and not filter_fn(agent, candidate):
                continue

            agent.neighbors.append(candidate)
            new_links.add(candidate)

        agent.n_neighbors = len(agent.neighbors)
    
def build_net(agents, level="buurt", k=4, filter_fn=None, return_nx=False):
    assign_local(agents, level=level)
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
