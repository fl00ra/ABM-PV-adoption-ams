import numpy as np
from agent import HouseholdAgent
from network import build_net
from data_loader import load_placeholder_data
from collections import defaultdict

class ABM:
    def __init__(self,
                 n_agents,
                 beta,
                 gamma,
                 policy_dict,
                 k_small_world,
                 net_level,
                 load_data=load_placeholder_data
                 ):

        self.n_agents = n_agents
        self.beta = beta
        self.gamma = gamma
        self.policy_dict = policy_dict
        self.k_small_world = k_small_world
        self.net_level = net_level

        self.agent_data = load_data()
        self.n_agents = min(self.n_agents, len(self.agent_data))
        self.agents = self._init_agents()

        # build the network
        self.nx_graph = build_net(
            self.agents,
            level=self.net_level,
            k=self.k_small_world,
            filter_fn=None,
            return_nx=True
        )

        for agent in self.agents:
            self.nx_graph.nodes[agent.id]["adopted"] = agent.adopted
            self.nx_graph.nodes[agent.id]["visible"] = agent.visible
            self.nx_graph.nodes[agent.id]["income"] = agent.income
            self.nx_graph.nodes[agent.id]["location"] = agent.location_code

        self._assign_targeting(self.agents)

        # initialize results storage
        self.results = {
            "adoption_rate": [],
            "targeted_adoption_rate": [],
            "visible_rate": [],
            "new_adopters": [],
        }


    def _init_agents(self):
        agents = []
        for d in self.agent_data:
            agent = HouseholdAgent(
                agent_id=d["id"],
                model=self,
                income=d["income"],
                lambda_loss_aversion=d["lambda_loss_aversion"],
                gamma=self.gamma,
                location_code=d["location_code"],
                energielabel=d["energielabel"],
                elek_usage=d["elek_usage"],
                elec_price=d["elec_price"],
                household_type=d.get("household_type", "single"),
                lihe=d.get("lihe", 0),
                lekwi=d.get("lekwi", 0),
                lihezlek=d.get("lihezlek", 0),
                adopted=d.get("adopted", False) 
            )
            agents.append(agent)
        return agents

    def _assign_targeting(self, agents):
        strategy = self.policy_dict.get("strategy_tag", "")

        if strategy == "fast_adoption":
            # choose 20% highest degree agents
            agents_sorted = sorted(agents, key=lambda a: len(a.neighbors), reverse=True)
            n_top = int(0.2 * len(agents))
            for i, agent in enumerate(agents_sorted):
                agent.is_targeted = (i < n_top)

        elif strategy == "support_vulnerable":
            # choose 20% lowest income + 20% highest elek agents
            agents_sorted = sorted(agents, key=lambda a: (a.income, -a.elek))
            n_target = int(0.2 * len(agents))
            for i, agent in enumerate(agents_sorted):
                agent.is_targeted = (i < n_target)

        elif strategy == "universal_nudge":
            # random
            for agent in agents:
                agent.is_targeted = (np.random.rand() < 0.2)

        elif strategy == "behavioral_first":
            # choose 20% highest Z value
            agents_with_Z = [(agent, agent.compute_Zi()) for agent in agents]
            agents_sorted = sorted(agents_with_Z, key=lambda tup: tup[1], reverse=True)
            n_top = int(0.2 * len(agents))
            for i, (agent, _) in enumerate(agents_sorted):
                agent.is_targeted = (i < n_top)

        elif strategy == "no_policy":
            for agent in agents:
                agent.is_targeted = False

    def step(self, t):
        for agent in self.agents:
            agent.step(t)

    def run(self, n_steps):
        for t in range(n_steps):
            self.step(t)
            self._record(t)

    def _record(self, t):
        n_adopted = sum(1 for a in self.agents if a.adopted)
        n_visible = sum(1 for a in self.agents if a.visible)

        # compute targeted adoption rate
        targeted = [a for a in self.agents if a.is_targeted]
        n_targeted = len(targeted)
        n_targeted_adopted = sum(1 for a in targeted if a.adopted)
        targeted_rate = n_targeted_adopted / n_targeted if n_targeted > 0 else 0

        if t == 0:
            new_adopters = n_adopted  # all adopters are new at first
        else:
            prev_adopted = self.results["adoption_rate"][-1] * self.n_agents
            new_adopters = n_adopted - int(prev_adopted)

        self.results["adoption_rate"].append(n_adopted / self.n_agents)
        self.results["visible_rate"].append(n_visible / self.n_agents)
        self.results.setdefault("targeted_adoption_rate", []).append(targeted_rate)
        self.results["new_adopters"].append(new_adopters)

    def get_results(self):
        return self.results
