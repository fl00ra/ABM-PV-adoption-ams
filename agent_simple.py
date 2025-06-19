import numpy as np
from config import THETA
from scipy.stats import truncnorm

class HouseholdAgent:
    def __init__(self, agent_id, model,
                 income,
                #  location_code,
                 energielabel=None, elek_usage=None, elec_price=0.4,
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
        self.adopted = adopted
        self.is_targeted = False
        self.policy_applied = False

        # spatial location 
        # self.location_code = location_code  # GWBCODEJJJJ
        # self.gemeente_code = location_code[:4]
        # self.wijk_code = location_code[4:6]
        # self.buurt_code = location_code[6:8]
        self.buurt_code = buurt_code  


        # electricity attributes
        self.elek = elek_usage
        self.elek_return = elek_return
        self.elec_price = elec_price
        self.energielabel = energielabel
        self.label_score = self._label_to_score(energielabel)

        self.household_type = household_type
        # self.household_type_vector = household_type_map.get(household_type, [0, 0, 0])
        self.bbihj = bbihj

        self.system_size = self._sample_system_size()
        self.pv_price = self._sample_pv_price()
        self.y_pv = self._sample_ypv()
        self.cost = self._compute_cost()
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
        self.social_weight = self._sample_social_weight()

        self.status = "active"  # or deliberate, exit
        self.steps_without_adoption = 0

        self.postcode6 = postcode6

        self.adoption_track = []
        self.history = []  

    def compute_Vi(self, timestep=0):
        """Vi = (Y * Gi - Ci) / θ"""
        # raw_V = self.gain - self.lambda_loss_aversion * self.cost
        if not hasattr(self, "gain") or self.gain is None:
            self.gain = self.model.compute_gain_fn(self, timestep)
        raw_V = self.Y * self.gain - self.cost
        return raw_V / THETA
        # return raw_V


    def compute_Si(self):
        adopted_weight = 0
        total_weight = 0

        for neighbor in self.neighbors:
            w = neighbor.social_weight
            if neighbor.adopted:
                adopted_weight += w
            total_weight += w

        return adopted_weight / total_weight if total_weight > 0 else 0


    # def _spatial_weight(self, neighbor):
    #     if self.location_code == neighbor.location_code:
    #         return 1.0  
    #     elif self.gemeente_code == neighbor.gemeente_code and self.wijk_code == neighbor.wijk_code:
    #         return 0.6  
    #     elif self.gemeente_code == neighbor.gemeente_code:
    #         return 0.3  
    #     else:
    #         return 0.1  
        

    
    def compute_beta0(self):
        """
        embed structural attributes into agent-specific intercept.
        """
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
        return np.random.normal(loc=-3.0, scale=0.5)




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


    
    def step(self, timestep=None):
        """
        behavioral logic with accumulation-based adoption (adding inertia mechanism.
        """

        if self.status == "exit":
            return  

        if self.adopted:
            return
        
        self.model.policy.apply_to(self, timestep)

        # self.gain = self.model.compute_gain_fn(self, timestep)

        prob, V, S, beta0 = self.compute_adoption_probability(timestep)
        self.history.append(prob)

        # # accumulate motivation with decay (λ = 0.8)
        # self.motivation_score = 0.8 * self.motivation_score + prob

        # # adoption decision by threshold
        # if self.motivation_score >= self.adoption_threshold:
        #     self.adopted = True
        #     self.adoption_time = timestep

        # self.adoption_track.append(self.adopted)

        print(f"[T={timestep}] Agent {self.id} | P={prob:.2f}, Beta0={beta0:.2f}, V={V:.2f}, S={S:.2f}, Score={self.motivation_score:.2f}")

        # Bernoulli trial for adoption
        if np.random.rand() < prob:
            self.adopted = True
            self.adoption_time = timestep

        if not self.adopted:
            self.steps_without_adoption += 1

            if self.status == "active" and self.steps_without_adoption >= 5:
                self.status = "deliberate"

            elif self.status == "deliberate":
                S = self.compute_Si()
                if S > 0.4 and np.random.rand() < 0.7:
                    self.status = "exit"
        else:
            self.steps_without_adoption = 0




    # def apply_policy(self, timestep=None):
    #     if self.adopted or self.policy_applied:
    #         return  

    #     policy = self.model.policy_dict
    #     strategy = policy.get("strategy_tag", "")

    #     if strategy == "reduce_cost":
    #         self.cost *= 0.8  

    #     elif strategy == "no_policy":
    #         pass

    #     self.policy_applied = True

    # def apply_policy(self, timestep=None):
    #     if self.adopted or self.policy_applied:
    #         return

    #     policy = self.model.policy 
    #     policy.apply_to(self, timestep)

    #     self.policy_applied = True


    def _label_to_score(self, label):
        mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        return mapping.get(label.upper(), 4)

    # # feed-in tariff may be added later
    # def _compute_gain(self, eta=0.9):
    #     return eta * self.elek * self.elec_price

    # def _compute_cost(self, C0=3000, alpha=0.1): #c0 will be updated later in the dataset
    #     normalized_score = (self.label_score - 1) / 6  # ∈[0, 1]
    #     return C0 * (1 + alpha * normalized_score)


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

    # def _compute_gain(self):
    #     """
    #     Gain = 
    #     if Ei_gen ≤ ELEK: ELEK * P_elek
    #     else: ELEK * P_elek + surplus * P_feed
    #     """
    #     Ei_gen = self.system_size * self.y_pv
    #     self.gen_electricity = Ei_gen

    #     elec_price = self.elec_price  # €/kWh
    #     feed_in_tariff = 0.10 

    #     if self.elek is None:
    #         self.elek = 2500  # fallback default

    #     if Ei_gen <= self.elek:
    #         return Ei_gen * elec_price
    #     else:
    #         saved = self.elek * elec_price
    #         surplus = Ei_gen - self.elek
    #         exported = surplus * feed_in_tariff
    #         return saved + exported

    def _infer_Y(self):
        base_Y = 3
        if self.income is not None:
            if self.income < 20000:
                base_Y -= 1
            elif self.income > 55000:
                base_Y += 2
        if self.lihe:
            base_Y -= 1.5
        return np.clip(base_Y + np.random.normal(0, 1), 2, 6)

    def _sample_social_weight(self):
        dist_params = {
            "with_kids": (1.2, 0.1),
            "couple_no_kids": (1.0, 0.1),
            "single": (0.8, 0.1),
            "single_parent": (0.9, 0.1)
        }
        mu, sigma = dist_params.get(self.household_type, (1.0, 0.1))
        return np.clip(np.random.normal(mu, sigma), 0.3, 2.0)