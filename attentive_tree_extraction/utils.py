import numpy as np
import torch
from scobi import Environment
from stable_baselines3 import PPO



def freeway_distance(p: np.ndarray,
                     q: np.ndarray,
                     lane_height: float = 15.0,   # pixels between lane centres
                     lane_penalty: float = 1e4,   # anything > screen height works
                     α: float = 0.05):            # tiebreak for same-lane cars
    """
p, q  are (x, y) coordinates in screen pixels.

• If q is in a **different lane**  →  distance = lane_penalty × |Δlane|
(so it is always considered “farther” than every car in my current lane)

• If q is in the **same lane**     →  distance = |Δy| + α·|Δx|
(Δx is O(160 px); pick α ≈ 0.03–0.08 so the horizontal term
is only 3–8 % as important as Δy)
    """
    # 1) which horizontal stripe (lane) are we in?
    lane_p = int(p[1] // lane_height)      # y is vertical in Freeway
    lane_q = int(q[1] // lane_height)

    if lane_p != lane_q:                   # car in another lane
        return lane_penalty * abs(lane_p - lane_q)

    # 2) same lane  →  vertical gap dominates, horizontal tie‑breaker
    return abs(p[1] - q[1]) + α * abs(p[0] - q[0])

def pong_distance(p, q):           # horizontal only
    return abs(p[0] - q[0])


class LogProbQ:
    def __init__(self, stochastic_pol: PPO, env: Environment):
        self.pol = stochastic_pol
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def q(self, s):
        s = torch.Tensor(s).to(self.device)
        s_repeat = s.repeat(self.env.action_space.n,1)
        with torch.no_grad():
            _, s_a_log_probs, _ = self.pol.policy.evaluate_actions(s_repeat, torch.arange(self.env.action_space.n).reshape(-1, 1).to(self.device))
        return s_a_log_probs

    def get_disagreement_cost(self, s):
        log_prob = self.q(s)
        return log_prob.mean() - log_prob.min()