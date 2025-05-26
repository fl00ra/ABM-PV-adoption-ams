import numpy as np
from config import household_type_map

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
        self.visible = False
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

        self.household_type = household_type  # e.g. "with_kids"
        self.household_type_vector = household_type_map.get(household_type, [0, 0, 0])
        self.bbihj = bbihj

        self.cost = self._compute_cost()
        self.gain = self._compute_gain()

        self.neighbors = []
        self.n_neighbors = 0
        self.adoption_time = None

        self.lihe = lihe
        self.lekwi = lekwi
        self.lihezlek = lihezlek

        self.history = []  

    def compute_Vi(self):
        """Vi(t) = Gi - λ * Ci"""
        # return self.gain - self.lambda_loss_aversion * self.cost
        return (self.gain - self.lambda_loss_aversion * self.cost) / self.cost

    
    def compute_Si(self):
        """S_i^visible(t) = (1 / n_i) * sum_j [adopted_j * σ_j]"""
        if self.n_neighbors == 0:
            return 0

        total_effect = 0
        for neighbor in self.neighbors:
            if neighbor.adopted:
                visibility_strength = 1.0 if getattr(neighbor, "visible", False) else 0.3
                total_effect += visibility_strength

        return total_effect / self.n_neighbors

    # def compute_Si(self):
    #     """S_i = n_adopted / n_total"""
    #     self.n_adopted_neighbors = sum(1 for neighbor in self.neighbors if neighbor.adopted)
    #     if self.n_neighbors == 0:
    #         return 0
    #     return self.n_adopted_neighbors / self.n_neighbors

    def compute_Zi(self):
        """Z_i = γ^T * X_i"""
        X_i = np.array([
            self.income / 1000,
            int(self.lihe),
            int(self.lekwi),
            int(self.lihezlek)
        ] + self.household_type_vector)
        return np.dot(self.gamma, X_i)
        #self.gamma = np.array([+1.5, -1.5, -1.0, -2.0, 0.5, 1.0, 0.2])

    def compute_adoption_probability(self):
        """P_i(t) = sigmoid(β₀ + β₁V + β₂S + β₃Z)"""
        V = self.compute_Vi()
        S = self.compute_Si()
        Z = self.compute_Zi()
        beta = self.model.beta  # read beta from model
        features = np.array([1, V, S, Z])  # 1: bias term beta0
        logit = np.dot(beta, features)
        prob = 1 / (1 + np.exp(-logit))

        # print(f"P= {probability:.2f}, V={V:.2f}, S={S:.2f}, Z={Z:.2f}")
        return prob, V, S, Z 
        # return prob

    def step(self, timestep=None):
        """
        behavioral logic for each timestep
        """
        self.apply_policy(timestep)

        if self.adopted:
            return
        
        prob, V, S, Z = self.compute_adoption_probability()
        # p = self.compute_adoption_probability()
        self.history.append(prob)

        print(f"[T={timestep}] Agent {self.id} | P={prob:.2f}, V={V:.2f}, S={S:.2f}, Z={Z:.2f}")


        # perform adoption with probability p (Bernoulli trial)
        # p_threshold = 0.3

        # if p >= p_threshold:
        #     if np.random.rand() < p:
        #         self.adopted = True
        #         self.adoption_time = timestep
        if np.random.rand() < prob:
            self.adopted = True
            self.adoption_time = timestep

        if self.is_targeted and self.adopted and not self.visible:
            self.visible = True

        # randomly visible to ensure spread
        elif not self.is_targeted and self.adopted and not self.visible:
            if np.random.rand() < 0.3:  
                self.visible = True
    
    
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
                # self.visible = True 

        elif strategy == "equity_first":
            if getattr(self, "is_targeted", False):
                self.cost *= 0.6  
                self.gain += 1500
                self.lambda_loss_aversion *= 0.75  
                #self.visible = True

        elif strategy == "universal_nudge":
            self.cost -= 1000 
            self.gain += 1000

        elif strategy == "behavioral_first":
            if getattr(self, "is_targeted", False):
                self.lambda_loss_aversion *= 0.7  
                # self.visible = True
        
        elif strategy == "no_policy":
            pass

        self.policy_applied = True 


    def _label_to_score(self, label):
        mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        return mapping.get(label.upper(), 4)

    def _compute_cost(self, C0=3000, alpha=0.1): #c0 will be updated later in the dataset
        normalized_score = (self.label_score - 1) / 6  # label:A=0, G=1
        return C0 * (1 + alpha * normalized_score)

    # feed-in tariff may be added later
    def _compute_gain(self, eta=0.9):
        return eta * self.elek * self.elec_price

