import numpy as np
from collections import deque

class KRSICalculator:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.history = {
            'E_eff': deque(maxlen=window_size),
            'CIW': deque(maxlen=window_size),
            'U_band': deque(maxlen=window_size),
            'Ren': deque(maxlen=window_size),
            'A': deque(maxlen=window_size),
            'R_rel': deque(maxlen=window_size),
            'R_rec': deque(maxlen=window_size),
            'R_perf': deque(maxlen=window_size),
            'A_adapt': deque(maxlen=window_size),
            'Q_work': deque(maxlen=window_size),
            'Q_det': deque(maxlen=window_size),
            'Q_blast': deque(maxlen=window_size),
            'Q_comp': deque(maxlen=window_size),
            'Q_rest': deque(maxlen=window_size)
        }
        self.raw_history = {
            'E_eff': deque(maxlen=window_size),
            'CIW': deque(maxlen=window_size)
        }
        self.max_e_eff_prev = 1.0
        self.max_ciw_prev = 1.0

    def _safe_std(self, q):
        std = np.std(q) if len(q) > 1 else 1.0
        return max(float(std), 0.05)

    def _get_adaptive_weights(self, keys):
        stds = [self._safe_std(self.history[k]) for k in keys]
        inv_stds = [1.0 / s for s in stds]
        sum_inv = sum(inv_stds)
        if sum_inv < 1e-5:
            return [1.0 / len(keys)] * len(keys)
        return [i / sum_inv for i in inv_stds]

    def compute(self, metrics):
        # 8.3 Sustainability Computation
        sm = metrics['sustainability']
        E_total = max(sm['e_total'], 1.0)
        W = sm['w']
        
        # Raw components
        E_eff = W / E_total
        CIW = W / (E_total * max(sm['ci'], 0.01))
        
        U_res = (sm['u_cpu'] + sm['u_mem'] + sm['u_sto']) / 3.0
        U_band = 1.0 - abs(U_res - sm['u_target']) / max(sm['u_target'], 0.1)
        U_band = max(0.0, min(1.0, U_band))
        
        Ren = max(0.0, min(1.0, sm['e_renewable'] / E_total))
        
        # Decorrelate and normalize
        self.raw_history['E_eff'].append(E_eff)
        self.raw_history['CIW'].append(CIW)
        
        if len(self.raw_history['E_eff']) < 20:
            new_max_e_eff = max(self.raw_history['E_eff'])
        else:
            new_max_e_eff = np.percentile(self.raw_history['E_eff'], 95)
            
        if len(self.raw_history['CIW']) < 20:
            new_max_ciw = max(self.raw_history['CIW'])
        else:
            new_max_ciw = np.percentile(self.raw_history['CIW'], 95)
            
        # Smoothing
        max_e_eff = 0.9 * self.max_e_eff_prev + 0.1 * new_max_e_eff
        max_ciw = 0.9 * self.max_ciw_prev + 0.1 * new_max_ciw
        
        self.max_e_eff_prev = max(max_e_eff, 1e-5)
        self.max_ciw_prev = max(max_ciw, 1e-5)
        
        norm_E_eff = max(0.0, min(1.0, E_eff / self.max_e_eff_prev))
        norm_CIW = max(0.0, min(1.0, CIW / self.max_ciw_prev))

        self.history['E_eff'].append(norm_E_eff)
        self.history['CIW'].append(norm_CIW)
        self.history['U_band'].append(U_band)
        self.history['Ren'].append(Ren)
        
        w_s = self._get_adaptive_weights(['E_eff', 'CIW', 'U_band', 'Ren'])
        S = w_s[0]*norm_E_eff + w_s[1]*norm_CIW + w_s[2]*U_band + w_s[3]*Ren
        S = max(0.0, min(1.0, S))
        
        # 8.6 Operational Resilience
        rm = metrics['resilience']
        A = max(0.0, min(1.0, rm['t_up'] / max(rm['t_obs'], 1.0)))
        R_rel = 1.0 if rm['n_f'] == 0 else max(0.0, min(1.0, 1.0 / (rm['n_f'] + 1)))
        R_rec = max(0.0, min(1.0, 1.0 - (rm['t_rec'] / max(rm['t_rec_max'], 1.0))))
        
        self.history['A'].append(A)
        self.history['R_rel'].append(R_rel)
        self.history['R_rec'].append(R_rec)
        w_op = self._get_adaptive_weights(['A', 'R_rel', 'R_rec'])
        R_op = w_op[0]*A + w_op[1]*R_rel + w_op[2]*R_rec
        R_op = max(0.0, min(1.0, R_op))
        
        # 8.7 Service Resilience
        R_perf = max(0.0, min(1.0, rm['p_min'] / max(rm['p_baseline'], 0.01)))
        A_adapt = max(0.0, min(1.0, rm['p_post'] / max(rm['p_baseline'], 0.01)))
        Q_work = max(0.0, min(1.0, rm['w_disruption'] / max(rm['w_baseline'], 0.01)))
        
        self.history['R_perf'].append(R_perf)
        self.history['A_adapt'].append(A_adapt)
        self.history['Q_work'].append(Q_work)
        w_srv = self._get_adaptive_weights(['R_perf', 'A_adapt', 'Q_work'])
        R_srv = w_srv[0]*R_perf + w_srv[1]*A_adapt + w_srv[2]*Q_work
        R_srv = max(0.0, min(1.0, R_srv))
        
        # 8.8 Adversarial Resilience
        Q_det = max(0.0, min(1.0, 1.0 - (rm['t_det'] / max(rm['t_det_max'], 1.0))))
        Q_blast = max(0.0, min(1.0, 1.0 - (rm['n_affected'] / max(rm['n_total'], 1.0))))
        Q_comp = max(0.0, min(1.0, 1.0 - rm['severity_score']))
        Q_rest = max(0.0, min(1.0, 1.0 - (rm['t_sec'] / max(rm['t_sec_max'], 1.0))))
        
        self.history['Q_det'].append(Q_det)
        self.history['Q_blast'].append(Q_blast)
        self.history['Q_comp'].append(Q_comp)
        self.history['Q_rest'].append(Q_rest)
        w_adv = self._get_adaptive_weights(['Q_det', 'Q_blast', 'Q_comp', 'Q_rest'])
        
        R_adv_raw = w_adv[0]*Q_det + w_adv[1]*Q_blast + w_adv[2]*Q_comp + w_adv[3]*Q_rest
        R_adv = max(0.0, min(1.0, rm['o_conf'] * R_adv_raw))
        
        # 8.9 Final Resilience (Soft-min with log-sum-exp trick)
        k = 5.0
        try:
            m = min(R_op, R_srv, R_adv)
            exp_sum = np.exp(-k * (R_op - m)) + np.exp(-k * (R_srv - m)) + np.exp(-k * (R_adv - m))
            R_soft_val = m - (np.log(exp_sum) / k)
            R_soft = R_soft_val / np.log(3)
            R = max(0.0, min(1.0, R_soft))
        except:
            R = min(R_op, R_srv, R_adv)
        
        # 8.10 Final KRSI
        KRSI = 0.0
        if R + S > 1e-5:
            KRSI = (2 * R * S) / (R + S)
            
        return max(0.0, min(1.0, KRSI)), S, R
