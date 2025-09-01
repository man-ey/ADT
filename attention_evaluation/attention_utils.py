import torch
import numpy as np
from torch import nn

class AtariDQNWrapper:
    def __init__(self, net):
        self.net = net
        self.device = next(net.parameters()).device

    # SB3-style API
    def predict(self, obs, deterministic=True):
        if isinstance(obs, np.ndarray):  # VecEnv gives np
            obs_t = torch.from_numpy(obs).to(self.device)
        else:
            obs_t = obs.to(self.device)
        if obs_t.dim() == 3:            # (4,84,84)  → (1,4,84,84)
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            q = self.net(obs_t)
        action = q.argmax(1).cpu().numpy()
        return action, None

class AtariNet(nn.Module):
    def __init__(self, action_no, distributional=False):
        super().__init__()
        self.action_no = action_no
        self.distributional = distributional

        # --- same layer names as in the checkpoint -----------------
        self.__features = nn.Sequential(                 # ← note the __
                                        nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
                                        nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                                        nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                                        )
        self.__head = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU(),
            nn.Linear(512, action_no * 51 if distributional else action_no),
        )
        # -----------------------------------------------------------

    def forward(self, x):
        x = x.float() / 255.0
        x = self.__features(x)
        x = x.view(x.size(0), -1)
        q = self.__head(x)
        if self.distributional:              # collapse C51 logits → Q
            q = q.view(-1, self.action_no, 51).mean(-1)
        return q



from gzip import GzipFile
from pathlib import Path

def load_atarinet(path, action_no, device="cpu"):
    net = AtariNet(action_no, distributional=("C51_" in path))
    with Path(path).open("rb") as f, GzipFile(fileobj=f) as gz:
        state = torch.load(gz, map_location=device)
    net.load_state_dict(state["estimator_state"])
    net.to(device).eval()
    return net


from ale_env import ALEClassic          # the same helper the original repo uses

def make_env_for_ckpt(game, record_dir=None):
    return ALEClassic(
        game,
        seed=torch.randint(100_000, ()).item(),
        sdl=False,                       # software renderer off → faster
        device="cpu",
        clip_rewards_val=False,
        record_dir=record_dir,
        #repeat_action_probability=0.25,  # or 0.0 if that’s what they used
        #terminal_on_life_loss=False,
    )