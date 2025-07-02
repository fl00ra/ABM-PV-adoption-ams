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

# def total_deg():
#     degree = int(round(np.random.lognormal(mean=np.log(7), sigma=0.5)))
#     return max(2, degree)

# # def assign_local(agents, level="buurt", local_ratio=0.7, use_income=False):
# #     def similarity(agent1, agent2):
# #         household_score = 1.0 if agent1.household_type == agent2.household_type else 0.3
# #         income_score = 1.0 if agent1.income_level == agent2.income_level else 0.3
# #         if use_income:
# #             return 0.4 * household_score + 0.6 * income_score
# #         else:
# #             return household_score


# def assign_local(agents, level="buurt", local_ratio=0.7, hh_weight=0.1, income_weight=0.9):
#     def similarity(agent1, agent2):
#         household_score = 1.0 if agent1.household_type == agent2.household_type else 0.3
#         income_score = 1.0 if agent1.income_level == agent2.income_level else 0.3
#         return hh_weight * household_score + income_weight * income_score

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

# def build_net(agents, level="buurt", filter_fn=None, return_nx=False, assign_social_weight=True, hh_weight=0.1, income_weight=0.9):
#     agent_k_map = assign_local(agents, level=level, hh_weight=hh_weight, income_weight=income_weight)
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
#             values = list(centrality.values())

#             min_val, max_val = min(values), max(values)
#             agent_map = {a.id: a for a in agents}
    
#             for agent_id, value in centrality.items():
#                 normalized = 0.5 + (value - min_val) / (max_val - min_val) if max_val > min_val else 1.0
#                 agent_map[agent_id].social_weight = normalized

#         # if assign_social_weight:
#         #     centrality = nx.eigenvector_centrality(G, max_iter=1000)
#         #     agent_map = {a.id: a for a in agents}
#         #     for agent_id, value in centrality.items():
#         #         agent_map[agent_id].social_weight = value

#         if return_nx:
#             return G

import random
from collections import defaultdict
import networkx as nx
import numpy as np

def group_by(agents, level="buurt"):
    """按地理层级分组"""
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
    """
    更现实的度分布：大多数人只有少量连接
    """
    # 80%的人有2-5个连接，20%的人有6-15个连接
    if random.random() < 0.8:
        return random.randint(2, 5)
    else:
        return random.randint(6, 15)

def assign_local(agents, level="buurt", local_ratio=0.9, hh_weight=0.1, income_weight=0.9):
    """
    创建高度局部化的网络，90%的连接在本地
    """
    def spatial_similarity(agent1, agent2):
        """计算空间相似性 - 同一区域内连接概率更高"""
        if agent1.buurt_code == agent2.buurt_code:
            return 1.0  # 同一buurt
        elif agent1.buurt_code[:6] == agent2.buurt_code[:6]:
            return 0.3  # 同一wijk但不同buurt
        elif agent1.buurt_code[:4] == agent2.buurt_code[:4]:
            return 0.1  # 同一gemeente但不同wijk
        else:
            return 0.01  # 完全不同区域
    
    def similarity(agent1, agent2):
        """综合相似性：空间距离是主要因素"""
        spatial_score = spatial_similarity(agent1, agent2)
        household_score = 1.0 if agent1.household_type == agent2.household_type else 0.5
        income_score = 1.0 if agent1.income_level == agent2.income_level else 0.5
        
        # 空间距离占主导地位
        social_score = hh_weight * household_score + income_weight * income_score
        return spatial_score * social_score

    groups = group_by(agents, level)
    agent_k_map = {}

    # 分配每个agent的本地和长距离连接数
    for agent in agents:
        k_total = total_deg()
        k_local = np.random.binomial(k_total, p=local_ratio)  # 90%是本地连接
        k_long = k_total - k_local
        agent_k_map[agent.id] = (k_local, k_long)

    # 建立本地连接
    for group in groups.values():
        for agent in group:
            others = [a for a in group if a.id != agent.id]
            k_local, _ = agent_k_map[agent.id]

            if len(others) > k_local:
                # 基于相似性的概率选择邻居
                weights = np.array([similarity(agent, other) for other in others])
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    neighbors = list(np.random.choice(others, size=k_local, replace=False, p=weights))
                else:
                    neighbors = random.sample(others, k=k_local)
            else:
                neighbors = others

            agent.neighbors = neighbors
            agent.n_neighbors = len(agent.neighbors)

    return agent_k_map

def add_long_links(agents, agent_k_map, filter_fn=None):
    """
    添加少量策略性的长距离连接，主要连接不同buurt的桥接节点
    """
    # 按buurt分组
    buurt_groups = defaultdict(list)
    for agent in agents:
        buurt_groups[agent.buurt_code].append(agent)
    
    # 计算每个buurt的中心节点（度数最高的几个）
    buurt_hubs = {}
    for buurt, members in buurt_groups.items():
        # 按当前邻居数排序，选择前20%作为潜在桥接节点
        sorted_members = sorted(members, key=lambda a: len(a.neighbors), reverse=True)
        n_hubs = max(1, len(sorted_members) // 5)
        buurt_hubs[buurt] = sorted_members[:n_hubs]
    
    # 为每个agent添加长距离连接
    for agent in agents:
        k_local, k_long = agent_k_map[agent.id]
        
        if k_long > 0:
            # 获取其他buurt的列表
            other_buurts = [b for b in buurt_groups.keys() if b != agent.buurt_code]
            
            if other_buurts:
                # 优先连接到地理上较近的buurt
                # 按wijk分组，优先同wijk不同buurt
                same_wijk_buurts = [b for b in other_buurts if b[:6] == agent.buurt_code[:6]]
                diff_wijk_buurts = [b for b in other_buurts if b[:6] != agent.buurt_code[:6]]
                
                # 70%概率连接同wijk，30%连接不同wijk
                new_links = []
                for _ in range(k_long):
                    if same_wijk_buurts and (not diff_wijk_buurts or random.random() < 0.7):
                        target_buurt = random.choice(same_wijk_buurts)
                    elif diff_wijk_buurts:
                        target_buurt = random.choice(diff_wijk_buurts)
                    else:
                        continue
                    
                    # 从目标buurt的hub中选择一个节点
                    candidates = buurt_hubs.get(target_buurt, buurt_groups[target_buurt])
                    if candidates:
                        # 避免重复连接
                        valid_candidates = [c for c in candidates 
                                          if c not in agent.neighbors and c.id != agent.id]
                        if valid_candidates:
                            chosen = random.choice(valid_candidates)
                            # 建立双向连接
                            agent.neighbors.append(chosen)
                            chosen.neighbors.append(agent)
                            new_links.append(chosen)
                
        agent.n_neighbors = len(agent.neighbors)

def build_net(agents, level="buurt", filter_fn=None, return_nx=False, 
             assign_social_weight=True, hh_weight=0.1, income_weight=0.9, local_ratio=0.9):
    """
    构建具有强地理聚类的网络
    """
    # 建立本地连接
    agent_k_map = assign_local(agents, level=level, local_ratio=local_ratio, 
                              hh_weight=hh_weight, income_weight=income_weight)
    
    # 添加长距离连接
    add_long_links(agents, agent_k_map, filter_fn=filter_fn)
    
    # 验证网络对称性
    for agent in agents:
        for neighbor in agent.neighbors[:]:  # 使用切片避免修改时的问题
            if agent not in neighbor.neighbors:
                neighbor.neighbors.append(agent)
    
    # 构建NetworkX图
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
            # 使用度中心性而不是特征向量中心性，更稳定
            degree_centrality = nx.degree_centrality(G)
            values = list(degree_centrality.values())
            
            # 标准化到[0.5, 2.0]范围
            if len(values) > 1:
                min_val, max_val = min(values), max(values)
                agent_map = {a.id: a for a in agents}
                
                for agent_id, value in degree_centrality.items():
                    if max_val > min_val:
                        # 映射到[0.5, 2.0]
                        normalized = 0.5 + 1.5 * (value - min_val) / (max_val - min_val)
                    else:
                        normalized = 1.0
                    agent_map[agent_id].social_weight = normalized
            else:
                for agent in agents:
                    agent.social_weight = 1.0
        
        # 打印网络统计信息
        if return_nx:
            avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
            clustering = nx.average_clustering(G)
            print(f"Network stats: Avg degree={avg_degree:.1f}, Clustering={clustering:.3f}")
            
            # 统计地理连接分布
            local_edges = 0
            same_wijk_edges = 0
            long_edges = 0
            
            for u, v in G.edges():
                agent_u = next(a for a in agents if a.id == u)
                agent_v = next(a for a in agents if a.id == v)
                
                if agent_u.buurt_code == agent_v.buurt_code:
                    local_edges += 1
                elif agent_u.buurt_code[:6] == agent_v.buurt_code[:6]:
                    same_wijk_edges += 1
                else:
                    long_edges += 1
            
            total_edges = len(G.edges())
            print(f"Edge distribution: Local={local_edges/total_edges:.1%}, "
                  f"Same wijk={same_wijk_edges/total_edges:.1%}, "
                  f"Long distance={long_edges/total_edges:.1%}")
            
            return G

# 添加辅助函数：计算考虑地理距离的社会影响
def compute_spatial_social_influence(agent):
    """
    计算考虑地理距离衰减的社会影响
    """
    adopted_weight = 0
    total_weight = 0
    
    for neighbor in agent.neighbors:
        # 基础权重
        w = neighbor.social_weight
        
        # 地理距离衰减
        if agent.buurt_code == neighbor.buurt_code:
            distance_factor = 1.0  # 同一buurt，全影响
        elif agent.buurt_code[:6] == neighbor.buurt_code[:6]:
            distance_factor = 0.5  # 同一wijk，影响减半
        elif agent.buurt_code[:4] == neighbor.buurt_code[:4]:
            distance_factor = 0.3  # 同一gemeente，影响很小
        else:
            distance_factor = 0.2  # 不同gemeente，影响最小
        
        w = w * distance_factor
        
        if neighbor.adopted:
            adopted_weight += w
        total_weight += w
    
    return adopted_weight / total_weight if total_weight > 0 else 0