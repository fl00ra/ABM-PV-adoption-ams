import numpy as np
from agent_simple import HouseholdAgent
from network_geo import build_net
from data_loader import load_amsterdam_data
from collections import defaultdict
from policy_intervention import Policy, compute_gain


class ABM:
    def __init__(self,
                 n_agents,
                 beta,
                 #policy_dict,
                 #feed_mode,
                 behavior_mode,
                 k_small_world,
                 net_level,
                 load_data=load_amsterdam_data
                 ):

        self.n_agents = n_agents
        self.beta = beta
        self.policy = Policy(elec_price=0.4, feed_mode="net_metering", behavior_mode=behavior_mode)
        # self.policy_dict = policy_dict
        self.k_small_world = k_small_world
        self.net_level = net_level

        self.agent_data = load_data()
        self.n_agents = min(self.n_agents, len(self.agent_data))
        self.agents = self._init_agents()
        # self.policy_object = Policy(elec_price=0.4, policy_type="net_metering")
        self.compute_gain_fn = lambda agent, t: compute_gain(agent, t, self.policy)


        self.nx_graph = build_net(
            self.agents,
            level=self.net_level,
            filter_fn=None,
            return_nx=True
        )

        for agent in self.agents:
            self.nx_graph.nodes[agent.id]["adopted"] = agent.adopted
            self.nx_graph.nodes[agent.id]["income"] = agent.income
            self.nx_graph.nodes[agent.id]["buurt_code"] = agent.buurt_code

        self.results = {
            "adoption_rate": [],
            "targeted_adoption_rate": [],
            "new_adopters": [],
        }

    def _init_agents(self):
        agents = []
        for d in self.agent_data:
            agent = HouseholdAgent(
                agent_id=d["id"],
                model=self,
                income=d["income"],
                energielabel=d.get("energielabel"),
                elek_usage=d.get("elek_usage"),
                elec_price=d.get("elec_price", 0.4),
                household_type=d.get("household_type", "single"),
                postcode6=d.get("postcode6"),
                buurt_code=d.get("buurt_code"),
                lihe=d.get("lihe", 0),
                adopted=d.get("adopted", False)
            )
            agents.append(agent)
        return agents


    def step(self, t):
        for agent in self.agents:
            agent.step(t)

    def run(self, n_steps):
        for t in range(n_steps):
            self.step(t)
            self._record(t)

    def _record(self, t):
        n_adopted = sum(1 for a in self.agents if a.adopted)

        targeted = [a for a in self.agents if a.is_targeted]
        n_targeted = len(targeted)
        n_targeted_adopted = sum(1 for a in targeted if a.adopted)
        targeted_rate = n_targeted_adopted / n_targeted if n_targeted > 0 else 0

        if t == 0:
            new_adopters = n_adopted
        else:
            prev_adopted = self.results["adoption_rate"][-1] * self.n_agents
            new_adopters = n_adopted - int(prev_adopted)

        self.results["adoption_rate"].append(n_adopted / self.n_agents)
        self.results.setdefault("targeted_adoption_rate", []).append(targeted_rate)
        self.results["new_adopters"].append(new_adopters)

    def get_results(self):
        return self.results
