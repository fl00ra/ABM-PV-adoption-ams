import numpy as np
from config import THETA
from scipy.stats import truncnorm

class HouseholdAgent:
    def __init__(self, agent_id, model,
                 income,
                #  location_code,
                 energielabel=None, elek_usage=None, elec_price=None,
                 elek_return=None, household_type=None,
                 postcode6=None, buurt_code=None,
                 lihe=None, 
                #  lekwi=None, lihezlek=None,
                 bbihj=None,
                 adopted=False
                 ):
        
        self.id = agent_id
        self.model = model 

        self.income = income
        self.income_level = self._categorize_income()
        self.adopted = adopted
        self.is_targeted = False
        # self.policy_applied = False
        # self._effective_cost = None

        # spatial location 
        # self.location_code = location_code  # GWBCODEJJJJ
        # self.gemeente_code = location_code[:4]
        # self.wijk_code = location_code[4:6]
        # self.buurt_code = location_code[6:8]
        self.buurt_code = buurt_code  


        # electricity attributes
        self.elek = elek_usage
        # self.elek_return = elek_return
        self.elec_price = elec_price
        self.energielabel = energielabel
        self.label_score = self._label_to_score(energielabel)

        self.household_type = household_type
        # self.household_type_vector = household_type_map.get(household_type, [0, 0, 0])
        # self.bbihj = bbihj

        self.system_size = self._sample_system_size()
        self.pv_price = self._sample_pv_price()
        self.y_pv = self._sample_ypv()
        self.base_cost = self._compute_cost()
        # self.gain = self._compute_gain()
        self.beta0 = self.compute_beta0()

        self.neighbors = []
        self.n_neighbors = 0
        self.adoption_time = None

        self.lihe = lihe
        # self.lekwi = lekwi
        # self.lihezlek = lihezlek

        self.Y = self._infer_Y()

        self.motivation_score = 0.0
        self.adoption_threshold = np.random.uniform(2.5, 4.0)
        # self.social_weight = self._sample_social_weight()
        self.social_weight = 1.0  

        self.status = "active" 
        self.steps_without_adoption = 0

        self.postcode6 = postcode6

        self.adoption_track = []
        self.history = []  

    def _categorize_income(self):   
        low_thres, high_thres = getattr(self.model, "income_quantiles", (20000, 50000))

        if self.income < low_thres:
            return "low"
        elif self.income > high_thres:
            return "high"
        else:
            return "mid"


    # def compute_Vi(self, timestep=0):
    #     """Vi = (Y * Gi - Ci) / θ"""
    #     # raw_V = self.gain - self.lambda_loss_aversion * self.cost
    #     if not hasattr(self, "gain") or self.gain is None:
    #         self.gain = self.model.compute_gain_fn(self, timestep)

    #     raw_V = self.Y * self.gain - self.cost
    #     return raw_V / THETA
    #     # return raw_V

    def compute_Vi(self, timestep=0):
        """Vi = (Y * Gi - Ci_effective) / θ"""
        self.gain = self.model.compute_gain_fn(self, timestep)
    
        effective_cost = self.model.policy.get_effective_cost(self, timestep)
    
        raw_V = self.Y * self.gain - effective_cost
        return raw_V / THETA


    def compute_Si(self):
        adopted_weight = 0
        total_weight = 0

        for neighbor in self.neighbors:
            w = neighbor.social_weight
            if neighbor.adopted:
                adopted_weight += w
            total_weight += w

        return adopted_weight / total_weight if total_weight > 0 else 0

        

    
    def compute_beta0(self):
        # """
        # embed structural attributes into agent-specific intercept.
        # """
        # base = -2.0
        # if self.income is not None:
        #     base += 0.0003 * (self.income - 30000)  # income centered around 30k
        # if self.label_score is not None:
        #     base += -0.05 * self.label_score  # worse label reduces base tendency
        # if self.lihe:
        #     base += -0.2
        # if self.lekwi:
        #     base += -0.2
        # if self.lihezlek:
        #     base += -0.3
        # return base

        """
        Sample beta₀ from predefined distributions to reflect different preference structures.
        This can be unimodal (homogeneous) or bimodal (polarized).
        """
        dist_type = getattr(self.model, "beta0_dist_type", "unimodal")

        if dist_type == "unimodal":
            return np.random.normal(loc=-4.5, scale=0.5)
    
        elif dist_type == "bimodal":
            if np.random.rand() < 0.5:
                return np.random.normal(loc=-4.0, scale=0.3)  # Low-preference group
            else:
                return np.random.normal(loc=-2.0, scale=0.3)  # High-preference group
    



    def compute_adoption_probability(self, timestep=0):
        """P_i(t) = sigmoid(β₀ᵢ + β₁V + β₂S)"""
        V = self.compute_Vi(timestep)
        S = self.compute_Si()
        beta = self.model.beta  # [β₁, β₂]

        beta0 = self.beta0
        features = np.array([V, S])
        logit = beta0 + np.dot(beta, features)
        prob = 1 / (1 + np.exp(-logit))

        return prob, V, S, beta0



    # def update_status(self, V, S):
    #     if self.status == "active":
    #         if np.random.rand() < min(1.0, self.steps_without_adoption / 6):
    #         # if self.steps_without_adoption > 5 and S < 0.3:
    #             self.status = "hesitant"
    #     elif self.status == "hesitant":
    #         if S > 0.8 and np.random.rand() < 0.4:
    #             self.status = "active"
    #             self.steps_without_adoption = 0
    #         elif np.random.rand() < min(1.0, self.steps_without_adoption / 15):
    #         # if self.steps_without_adoption >= 15:
    #             self.status = "exit"



    def step(self, timestep=None):
        if self.adopted:
            return

        # # self.model.policy.apply_to(self, timestep)
        # if not self.policy_applied:
        #     self.model.policy.apply_to(self, timestep)

        prob, V, S, beta0 = self.compute_adoption_probability(timestep)
        self.history.append(prob)


        # if self.status == "active":
        #     if np.random.rand() < prob:
        #         self.adopted = True
        #         self.adoption_time = timestep
        # elif self.status == "hesitant":
        #     if np.random.rand() < prob * 0.6:  
        #         self.adopted = True
        #         self.adoption_time = timestep

        if np.random.rand() < prob:
            self.adopted = True
            self.adoption_time = timestep

        self.adoption_track.append(self.adopted)

        # if not self.adopted:
        #     self.steps_without_adoption += 1
        #     self.update_status(V, S)

        print(f"[T={timestep}] Agent {self.id} | P={prob:.2f}, V={V:.2f}, S={S:.2f}, β0={beta0:.2f}, Status={self.status}")


        #     # if self.status == "active" and self.steps_without_adoption >= 5:
        #     #     self.status = "hesitant"

        #     # elif self.status == "hesitant":
        #     #     S = self.compute_Si()
        #     #     if S > 0.4 and np.random.rand() < 0.7:
        #     #         self.status = "exit"
                



    def _label_to_score(self, label):
        mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        return mapping.get(label.upper(), 4)


    def _sample_system_size(self):
        # Truncated lognormal: ln(s_i) ~ N(μ=1.28, σ=0.47), max 10 kWp
        mu, sigma = 1.28, 0.47
        upper = 10
        s_samples = np.random.lognormal(mu, sigma, 1000)
        s_samples = s_samples[s_samples <= upper]
        return np.random.choice(s_samples) if len(s_samples) > 0 else 4.0

    def _sample_pv_price(self):
        return np.random.triangular(1100, 1300, 1500)

    def _sample_ypv(self):
        return np.random.triangular(850, 900, 950)

    def _compute_cost(self):
        """
        Cost = s_i * P_pv
        where s_i ~ Truncated Lognormal, P_pv ~ Triangular
        """
        return self.system_size * self.pv_price


    def _infer_Y(self):
        base_Y = 4
        if self.income is not None:
            if self.income < 20000:
                base_Y -= 1
            elif self.income > 55000:
                base_Y += 2
        if self.lihe:
            base_Y -= 1
        return np.clip(base_Y + np.random.normal(0, 0.5), 2, 6)

    # def _sample_social_weight(self):
    #     dist_params = {
    #         "with_kids": (1.2, 0.1),
    #         "couple_no_kids": (1.0, 0.1),
    #         "single": (0.8, 0.1),
    #         "single_parent": (0.9, 0.1)
    #     }
    #     mu, sigma = dist_params.get(self.household_type, (1.0, 0.1))
    #     return np.clip(np.random.normal(mu, sigma), 0.5, 1.5)

