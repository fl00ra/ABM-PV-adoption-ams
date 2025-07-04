from data_loader import dynamic_elec_price

class Policy:
    def __init__(self, behavior_mode="universal_nudge", enable_feed_change=True):
        # self.elec_price = elec_price
        self.behavior_mode = behavior_mode
        self.enable_feed_change = enable_feed_change

        self.policy_params = {
            "universal_nudge": {
                "discount_rate": 0.21,  
                "start_time": 0,
                "end_time": None
            },
            "energy_subsidies": {
                "base_amount": 800,
                "decline_rate": 0,  
                "eligibility": lambda agent: agent.lihe == 1
            },
            # "progressive_subsidy": {
            #     "low": {"amount": 1200, "decline_after": 10, "final_amount": 800},
            #     "mid": {"amount": 600, "decline_after": 10, "final_amount": 400},
            #     "high": {"amount": 0, "decline_after": None, "final_amount": 0}
            # },
            # "dynamic_subsidy": {
            #     "initial_rate": 0.3,
            #     "decline_per_step": 0.02,
            #     "minimum_rate": 0.05
            # },
            # "time_limited_subsidy": {
            #     "discount_rate": 0.3,  
            #     "end_time": 10,        
            #     "phase_out": False    
            # }
            "time_limited_subsidy": {
                "phases": [
                    {"start": 0, "end": 5, "discount": 0.35},    
                    {"start": 5, "end": 10, "discount": 0.25},   
                    {"start": 10, "end": 15, "discount": 0.1},  
                ],
                "default_discount": 0.0
            }

        }
    def get_feed_in_tariff(self, timestep, current_elec_price=None):

        if not self.enable_feed_change:
            return 0.0  

        if current_elec_price is None:
            from data_loader import dynamic_elec_price
            current_elec_price = dynamic_elec_price(timestep)

        # stage 1: net metering (early)
        if timestep < 3:
            return current_elec_price

        # stage 2: declining subsidy
        elif timestep < 7:
            reduction_factor = max(0.05, 1.0 - 0.09 * (timestep - 5))
            return current_elec_price * reduction_factor

        # stage 3: fixed market price
        else:
            return 0.10



    def get_effective_cost(self, agent, timestep):
        """
        Calculate the effective cost for an agent based on the current policy and timestep.
        """
        base_cost = agent.base_cost
        
        if self.behavior_mode == "universal_nudge":
            params = self.policy_params["universal_nudge"]
            if params["end_time"] is None or timestep <= params["end_time"]:
                return base_cost * (1 - params["discount_rate"])
            else:
                return base_cost
                
        elif self.behavior_mode == "energy_subsidies":
            params = self.policy_params["energy_subsidies"]
            if params["eligibility"](agent):
                subsidy = params["base_amount"] * (1 - params["decline_rate"] * timestep)
                subsidy = max(0, subsidy)  
                return max(0, base_cost - subsidy)
            else:
                return base_cost
            
        # elif self.behavior_mode == "progressive_subsidy":
        #     income_level = agent.income_level
        #     if income_level in self.policy_params["progressive_subsidy"]:
        #         params = self.policy_params["progressive_subsidy"][income_level]
        #         if params["decline_after"] and timestep >= params["decline_after"]:
        #             subsidy = params["final_amount"]
        #         else:
        #             subsidy = params["amount"]
        #         return max(0, base_cost - subsidy)
        #     else:
        #         return base_cost
                
        # elif self.behavior_mode == "dynamic_subsidy":
        #     params = self.policy_params["dynamic_subsidy"]
        #     current_rate = max(
        #         params["minimum_rate"],
        #         params["initial_rate"] - params["decline_per_step"] * timestep
        #     )
        #     return base_cost * (1 - current_rate)
            
        # elif self.behavior_mode == "time_limited_subsidy":
        #     params = self.policy_params["time_limited_subsidy"]
            
        #     if timestep < params["end_time"]:
        #         return base_cost * (1 - params["discount_rate"])
        #     elif params["phase_out"] and timestep < params["end_time"] + 5:
        #         phase_out_steps = timestep - params["end_time"]
        #         reduced_discount = params["discount_rate"] * (1 - phase_out_steps / 5)
        #         return base_cost * (1 - max(0, reduced_discount))
        #     else:
        #         return base_cost
        elif self.behavior_mode == "time_limited_subsidy":
            params = self.policy_params["time_limited_subsidy"]
    
            discount = params["default_discount"]
            for phase in params["phases"]:
                if phase["start"] <= timestep < phase["end"]:
                    discount = phase["discount"]
                    break
    
            return base_cost * (1 - discount)
                
        elif self.behavior_mode == "no_policy":
            return base_cost
            
        return base_cost



# def compute_gain(agent, timestep, policy: Policy):
#     Ei_gen = agent.system_size * agent.y_pv
#     E_use = min(Ei_gen, agent.elek)
#     E_export = max(0, Ei_gen - agent.elek)

#     p_elek = agent.elec_price
#     p_feed = policy.get_feed_in_tariff(timestep)

#     saving = E_use * p_elek
#     export = E_export * p_feed

#     return saving + export



# def compute_gain(agent, timestep, policy = Policy, T=25, r=0.05):
#     """
#     Compute Net Present Value (NPV) of PV system gains over T timesteps.
#     """
#     Ei_gen = agent.system_size * agent.y_pv      # annual generation (kWh)
#     E_use = min(Ei_gen, agent.elek)              # self-consumed
#     E_export = max(0, Ei_gen - agent.elek)       # fed into grid
#     p_elek = agent.elec_price                    # current retail price

#     npv = 0.0
#     for t in range(T):
#         feed_price = policy.get_feed_in_tariff(t)  # dynamic feed-in price
#         saving = E_use * p_elek
#         export = E_export * feed_price
#         cashflow = saving + export
#         npv += cashflow / ((1 + r) ** t)  # discount future gains

#     return npv

# def compute_gain(agent, timestep, policy: Policy, T=20):
#     """
#     Compute Net Present Value (NPV) of PV system gains over T years.
#     """
#     Ei_gen = agent.system_size * agent.y_pv      # annual generation (kWh)
#     E_use = min(Ei_gen, agent.elek)              # self-consumed
#     E_export = max(0, Ei_gen - agent.elek)       # fed into grid
#     p_elek = agent.elec_price                    # current retail price
    
#     discount_rates = {
#         "low": 0.08,   # 低收入群体折现率更高（更看重短期收益）
#         "mid": 0.05,   # 中等收入群体
#         "high": 0.03   # 高收入群体折现率更低（更看重长期收益）
#     }
    
#     r = discount_rates.get(agent.income_level, 0.05)  # 默认5%
    
#     npv = 0.0
#     for t in range(T):
#         # 获取未来t年的上网电价（相对于当前timestep）
#         future_timestep = timestep + t
#         feed_price = policy.get_feed_in_tariff(future_timestep)
        
#         # 计算年度现金流
#         saving = E_use * p_elek      # 节省的电费
#         export = E_export * feed_price  # 售电收入
#         annual_cashflow = saving + export
        
#         # 折现到现值
#         npv += annual_cashflow / ((1 + r) ** t)
    
#     return npv


def compute_gain(agent, timestep, policy: Policy, T=20):

    degradation_rate = 0.005  # 0.5% annual degradation

  
    base_Ei_gen = agent.system_size * agent.y_pv 

    # Ei_gen = Ei_gen * (1 - degradation_rate) ** t
    # E_use = min(Ei_gen, agent.elek)
    # E_export = max(0, Ei_gen - agent.elek)
    current_elec_price = agent.get_current_elec_price(timestep)
    
    discount_rates = {
        "low": 0.08,
        "mid": 0.05,
        "high": 0.03
    }
    r = discount_rates.get(agent.income_level, 0.07)
    
    # maintenance cost
    annual_maintenance = agent.system_size * 15  # 15euro/kWp/year
    inverter_replacement = agent.system_size * 300  # 300euro/kWp
    
    npv = 0.0
    
    for t in range(T):
        future_timestep = timestep + t
        
        future_elec_price = agent.get_current_elec_price(future_timestep)

        feed_price = policy.get_feed_in_tariff(future_timestep)

        Ei_gen_year = base_Ei_gen * (1 - degradation_rate) ** t
        
        E_use = min(Ei_gen_year, agent.elek)
        E_export = max(0, Ei_gen_year - agent.elek)
        
        # annual gain
        saving = E_use * future_elec_price
        export = E_export * feed_price
        annual_revenue = saving + export
        
        # annual maintenance cost
        if t < 10:
            maintenance_cost = annual_maintenance
        elif t < 15:
            maintenance_cost = annual_maintenance * 1.2
        else:
            maintenance_cost = annual_maintenance * 1.5
        
        net_cashflow = annual_revenue - maintenance_cost
        
        # inverter replacement cost
        if t == 12:
            net_cashflow -= inverter_replacement
            
        
        npv += net_cashflow / ((1 + r) ** t)
    
    return npv  