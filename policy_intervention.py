class Policy:
    def __init__(self, elec_price=0.4, feed_mode = "net_metering", behavior_mode="universal_nudge"):
        self.elec_price = elec_price
        self.behavior_mode = behavior_mode
        self.feed_mode = feed_mode 

    def get_feed_in_tariff(self, timestep):
        if self.feed_mode == "net_metering" and timestep < 5:
            return self.elec_price
        elif self.feed_mode == "declining":
            return max(0.05, self.elec_price * (0.9 ** (timestep)))
        elif self.feed_mode == "fixed_market":
            return 0.10
        else:
            return 0.0
        
    # def apply_behavioral_intervention(self, agent, timestep):

    #     if self.strategy_tag == "behavioral_push":
    #         if agent.status == "deliberate" and timestep >= 2:
    #             agent.elek *= 1.1  
    #         elif agent.status == "active" and timestep >= 3:
    #             agent.elek *= 1.05

    def apply_to(self, agent, timestep):


        # if self.policy_type == "awareness_campaign":
        #     agent.Y += 1.0  
        #     agent.adoption_threshold *= 0.85

        # elif self.policy_type == "support_vulnerable":
        #     if getattr(agent, "lihe", False):
        #         agent.cost *= 0.8

        if self.behavior_mode == "universal_nudge":
            agent.cost *= 0.79

        elif self.feed_mode != "net_metering":
            if self.behavior_mode == "behavioral_push":
                if agent.status == "deliberate" and timestep >= 2:
                    agent.elek *= 1.1
                elif agent.status == "active" and timestep >= 3:
                    agent.elek *= 1.05


        elif self.behavior_mode == "no_policy":
            pass    


        agent.policy_applied = True




def compute_gain(agent, year, policy: Policy):
    Ei_gen = agent.system_size * agent.y_pv
    E_use = min(Ei_gen, agent.elek)
    E_export = max(0, Ei_gen - agent.elek)

    p_elek = agent.elec_price
    p_feed = policy.get_feed_in_tariff(year)

    saving = E_use * p_elek
    export = E_export * p_feed

    return saving + export



