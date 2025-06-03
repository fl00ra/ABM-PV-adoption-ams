import numpy as np
from config import household_type_map, THETA
from scipy.stats import truncnorm

class HouseholdAgent:
    def __init__(self, agent_id, model,
                 income,
                 lambda_loss_aversion, gamma,
                 location_code,
                 energielabel=None, elek_usage=None, elec_price=0.4,
                 elek_return=None, household_type=None,
                 lihe=None, lekwi=None, lihezlek=None, bbihj=None,
                 adopted=False):
        
        self.id = agent_id
        self.model = model 

        self.income = income
        self.lambda_loss_aversion = lambda_loss_aversion
        self.gamma = gamma
        # self.Z = Z_vector
        self.adopted = adopted
        self.is_targeted = False
        self.policy_applied = False

        # spatial location 
        self.location_code = location_code  # GWBCODEJJJJ
        self.gemeente_code = location_code[:4]
        self.wijk_code = location_code[4:6]
        self.buurt_code = location_code[6:8]

        # electricity attributes
        self.elek = elek_usage
        self.elek_return = elek_return
        self.elec_price = elec_price
        self.energielabel = energielabel
        self.label_score = self._label_to_score(energielabel)

        self.household_type = household_type
        self.household_type_vector = household_type_map.get(household_type, [0, 0, 0])
        self.bbihj = bbihj

        self.system_size = self._sample_system_size()
        self.pv_price = self._sample_pv_price()
        self.y_pv = self._sample_ypv()
        self.cost = self._compute_cost()
        self.gain = self._compute_gain()

        self.neighbors = []
        self.n_neighbors = 0
        self.adoption_time = None

        self.lihe = lihe
        self.lekwi = lekwi
        self.lihezlek = lihezlek

        self.inertia = np.random.beta(2, 5)

        self.history = []  

    def compute_Vi(self):
        """Vi = (Gi - λ * Ci) / θ"""
        # return self.gain - self.lambda_loss_aversion * self.cost
        raw_V = self.gain - self.lambda_loss_aversion * self.cost
        return raw_V / THETA


    def compute_Si(self):
        """S_i = n_adopted / n_total"""
        self.n_adopted_neighbors = sum(1 for neighbor in self.neighbors if neighbor.adopted)
        if self.n_neighbors == 0:
            return 0
        return self.n_adopted_neighbors / self.n_neighbors
    
    def compute_beta0(self):
        """
        Embed structural attributes into agent-specific intercept.
        """
        base = 0
        if self.income is not None:
            base += 0.0001 * (self.income - 30000)  # income centered around 30k
        if self.label_score is not None:
            base += -0.1 * self.label_score  # worse label reduces base tendency
        if self.lihe:
            base += -0.5
        if self.lekwi:
            base += -0.3
        if self.lihezlek:
            base += -0.3
        return base



    def compute_adoption_probability(self):
        """P_i(t) = sigmoid(β₀ᵢ + β₁V + β₂S)"""
        V = self.compute_Vi()
        S = self.compute_Si()
        beta = self.model.beta  # assume shape: [β₁, β₂]

        beta0_i = self.compute_beta0()
        features = np.array([V, S])
        logit = beta0_i + np.dot(beta, features)
        prob = 1 / (1 + np.exp(-logit))

        return prob, V, S

    
    def step(self, timestep=None):
            """
            behavioral logic for each timestep, (adding inertia mechanism.
            """
            self.apply_policy(timestep)

            if self.adopted:
                return

            prob, V, S = self.compute_adoption_probability()
            self.history.append(prob)

            # reduce the probability by inertia
            adjusted_prob = max(0, prob - self.inertia)

            # Bernoulli trial for adoption
            if np.random.rand() < adjusted_prob:
                self.adopted = True
                self.adoption_time = timestep

            print(f"[T={timestep}] Agent {self.id} | P={prob:.2f}, V={V:.2f}, S={S:.2f}")

    
    def apply_policy(self, timestep=None):
        if self.adopted or self.policy_applied:
            return  
    
        policy = self.model.policy_dict
        strategy = policy.get("strategy_tag", "")

    # feed-in tariff may be added later

        if strategy == "fast_adoption":
            if getattr(self, "is_targeted", False):
                self.cost *= 0.8  # reduce Ci
                self.gain += 2000  # feed-in tariff

        elif strategy == "support_vulnerable":
            if getattr(self, "is_targeted", False):
                self.cost *= 0.6  
                self.gain += 1500
                self.lambda_loss_aversion *= 0.75  

        elif strategy == "universal_nudge":
            self.cost -= 500
            self.gain += 500

        elif strategy == "behavioral_first":
            if getattr(self, "is_targeted", False):
                self.lambda_loss_aversion *= 0.7  
                self.inertia *= 0.4 
        
        elif strategy == "no_policy":
            pass

        self.policy_applied = True 


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
        lower, upper = 0, 10
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

    def _compute_gain(self):
        """
        Gain = 
        if Ei_gen ≤ ELEK: ELEK * P_elek
        else: ELEK * P_elek + surplus * P_feed
        """
        Ei_gen = self.system_size * self.y_pv
        self.gen_electricity = Ei_gen

        elec_price = self.elec_price  # €/kWh
        feed_in_tariff = 0.10 

        if self.elek is None:
            self.elek = 2500  # fallback default

        if Ei_gen <= self.elek:
            return Ei_gen * elec_price
        else:
            saved = self.elek * elec_price
            surplus = Ei_gen - self.elek
            exported = surplus * feed_in_tariff
            return saved + exported
