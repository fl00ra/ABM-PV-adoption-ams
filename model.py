import numpy as np
from agent import HouseholdAgent
from network_geo import build_net
from data_loader import load_amsterdam_data, dynamic_elec_price, get_income_quantiles
from collections import defaultdict
from policy_intervention import Policy, compute_gain


class ABM:
    def __init__(self,
                 n_agents,
                 beta,
                 behavior_mode,
                 k_small_world,
                 net_level,
                 beta0_dist_type="unimodal",
                 enable_feed_change=True,
                 load_data=load_amsterdam_data
                 ):

        self.n_agents = n_agents
        self.beta = beta
        self.policy = Policy(behavior_mode=behavior_mode, enable_feed_change=enable_feed_change)
        self.k_small_world = k_small_world
        self.net_level = net_level
        self.beta0_dist_type = beta0_dist_type 

        self.agent_data = load_data()
        self.income_quantiles = get_income_quantiles(self.agent_data) 
        self.n_agents = min(self.n_agents, len(self.agent_data))
        self.agents = self._init_agents()
        self.compute_gain_fn = lambda agent, t: compute_gain(agent, t, self.policy)



        self.nx_graph = build_net(
            self.agents,
            level=self.net_level,
            filter_fn=None,
            return_nx=True,
            assign_social_weight=True,
            hh_weight= 0.1,
            income_weight= 0.9
        )

        for agent in self.agents:
            self.nx_graph.nodes[agent.id]["adopted"] = agent.adopted
            self.nx_graph.nodes[agent.id]["income"] = agent.income
            self.nx_graph.nodes[agent.id]["buurt_code"] = agent.buurt_code

        self.results = {
            "adoption_rate": [],
            "new_adopters": [],
            "group_adoption": defaultdict(list),
            "status_transitions": defaultdict(lambda: defaultdict(int)),
            "distributions": {
                "V": [],
                "S": [],
                "P": []
            },
            "elec_price": []

        }
    def get_dynamic_elec_price(self, timestep):
        return dynamic_elec_price(timestep)
    
    def _init_agents(self):
        agents = []
        for d in self.agent_data:
            agent = HouseholdAgent(
                agent_id=d["id"],
                model=self,
                income=d["income"],
                energielabel=d.get("energielabel"),
                elek_usage=d.get("elek_usage"),
                elec_price=None,
                household_type=d.get("household_type", "single"),
                postcode6=d.get("postcode6"),
                buurt_code=d.get("buurt_code"),
                lihe=d.get("lihe", 0),
                adopted=d.get("adopted", False)
            )
            agents.append(agent)

        self.beta0_values = [agent.beta0 for agent in agents]

        return agents


    def step(self, t):
        for agent in self.agents:
            agent.step(t)

    def run(self, n_steps):
        for t in range(n_steps):
            self.step(t)
            self._record(t)
            
            if t % 5 == 0:
                print(f"\n[Time {t}] Policy: {self.policy.behavior_mode}")
                print(f"Adoption rate: {self.results['adoption_rate'][-1]:.2%}")

    def change_policy(self, new_behavior_mode, timestep):
        """
        Change the policy behavior mode at a specific timestep.
        """
        self.policy.behavior_mode = new_behavior_mode
        print(f"\n[Policy Change at T={timestep}] New policy: {new_behavior_mode}")


    def _record(self, t):
        n_adopted = sum(1 for a in self.agents if a.adopted)

        if t == 0:
            new_adopters = n_adopted
        else:
            prev_adopted = self.results["adoption_rate"][-1] * self.n_agents
            new_adopters = n_adopted - int(prev_adopted)

        self.results["adoption_rate"].append(n_adopted / self.n_agents)
        self.results["new_adopters"].append(new_adopters)

        group_map = defaultdict(list)
        for a in self.agents:
            group_map[a.household_type].append(a)
        for g, ags in group_map.items():
            self.results["group_adoption"][g].append(sum(1 for a in ags if a.adopted) / len(ags))

        status_counts = defaultdict(int)
        for a in self.agents:
            status_counts[a.status] += 1
        for s, count in status_counts.items():
            self.results["status_transitions"][s][t] = count / self.n_agents

        V_vals, S_vals, P_vals = [], [], []
        for a in self.agents:
            p, V, S, _ = a.compute_adoption_probability(timestep=t)
            V_vals.append(V)
            S_vals.append(S)
            P_vals.append(p)

        self.results["distributions"]["V"].append(V_vals)
        self.results["distributions"]["S"].append(S_vals)
        self.results["distributions"]["P"].append(P_vals)

    def get_results(self):
        return self.results
