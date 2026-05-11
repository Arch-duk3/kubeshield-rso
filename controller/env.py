import requests
import numpy as np
import time

class K8sEnv:
    def __init__(self, url="http://localhost:8080"):
        self.url = url
        self.krsi_calc = None
        self.R_min = 0.6
        self.R_critical = 0.3
        self.action_space = 9
        self.state_dim = 12
        self.prev_action = 0
        self.prev_R = 1.0
        self.cooldown = 0
        self.max_nodes = 10
        self.max_rep = 20
        self.max_energy = 200.0 * self.max_nodes
        self.sla_latency = 50.0 # ms threshold
        self.last_raw_state = {'n_rep': 3, 'n_nodes': 3, 'p_sched': 0}
        
    def reset(self):
        try:
            requests.get(f"{self.url}/reset", timeout=5)
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to simulator: {e}")
            
        from krsi import KRSICalculator
        self.krsi_calc = KRSICalculator()
        self.prev_action = 0
        self.prev_R = 1.0
        self.cooldown = 0
        return self.step(0)[0]
        
    def step(self, action):
        resp = requests.post(f"{self.url}/step", json={"action": int(action)}).json()
        KRSI, S, R = self.krsi_calc.compute(resp)
        
        delta_u = 1 if action != self.prev_action and action != 0 else 0
        self.prev_action = action
        
        delta_R = R - self.prev_R
        self.prev_R = R
        
        sm = resp['state']
        self.last_raw_state = sm
        
        # State normalization
        norm_e = max(0.0, min(1.0, sm['e'] / self.max_energy))
        norm_l = max(0.0, min(1.0, sm['l'] / (self.sla_latency * 2.0)))
        norm_n_rep = max(0.0, min(1.0, sm['n_rep'] / self.max_rep))
        norm_n_nodes = max(0.0, min(1.0, sm['n_nodes'] / self.max_nodes))
        
        sched_val = -1.0 if sm['p_sched'] == 0 else 1.0
        
        state = np.array([
            sm['u_cpu'], sm['u_mem'], sm['u_sto'], 
            norm_e, R, S, delta_R, norm_l,
            max(0.0, min(1.0, sm['f'] / 10.0)), norm_n_rep, norm_n_nodes, sched_val
        ], dtype=np.float32)
        
        # Reward function
        lmbda = 2.0
        gamma = 0.1
        eta = 1.5
        
        SLA_violation = max(0.0, (sm['l'] - self.sla_latency) / self.sla_latency)
        penalty = max(0.0, self.R_min - R)
        
        reward = KRSI - (lmbda * penalty) - (gamma * delta_u) - (eta * SLA_violation)
        
        # Task 7: Reward shaping
        if action == self.prev_action and action != 0:
            reward += 0.1
            
        # Task 2: Replace reward clipping with tanh
        reward = 5.0 * np.tanh(float(reward) / 5.0)
        
        # Task 2: Episode termination
        done = bool(R < 0.2 or sm['l'] > 3 * self.sla_latency)
            
        info = {
            'KRSI': KRSI, 'S': S, 'R': R, 
            'action': action, 'n_rep': sm['n_rep'], 
            'n_nodes': sm['n_nodes'], 'latency': sm['l'], 
            'energy': sm['e'], 'raw': resp
        }
        
        if self.cooldown > 0:
            self.cooldown -= 1
            
        return state, reward, done, info
        
    def heuristic_action(self, state, R):
        if self.cooldown > 0:
            return 0 # NO_OP
            
        self.cooldown = 2 # Hysteresis/Cooldown
        n_rep = self.last_raw_state['n_rep']
        n_nodes = self.last_raw_state['n_nodes']
        p_sched = self.last_raw_state['p_sched']
        latency = self.last_raw_state['l']
        energy = self.last_raw_state['e']
        
        if R < self.R_min:
            if n_rep < n_nodes * 2:
                return 1 # SCALE_OUT_SMALL
            elif p_sched == 0:
                return 6 # SET_SPREAD
            elif n_nodes < self.max_nodes:
                return 3 # ADD_NODE
            else:
                return 8 # SCALE_THRESHOLD_DOWN
        else:
            if latency > self.sla_latency:
                if n_rep < self.max_rep:
                    return 1
                elif n_nodes < self.max_nodes:
                    return 3
                return 0
            elif energy > self.max_energy * 0.5:
                if n_rep > n_nodes:
                    return 2
                elif n_nodes > 1:
                    return 4
                return 0
            else:
                if p_sched == 1:
                    return 5 # SET_BINPACK
                return 7 # SCALE_THRESHOLD_UP
