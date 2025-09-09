from stable_baselines3 import PPO
from torch.backends.cudnn import deterministic

from scobi import Environment
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import numpy as np
import re
from collections import defaultdict
import gymnasium
from hackatari.core import HackAtari

# Custom environment used for training of tennis agenst in SCoBots
class PressOnReset(gymnasium.Wrapper):
    def __init__(self, env, keywords=("UP", "UPFIRE", "FIRE"), n_presses=8):
        super().__init__(env)
        self.keywords = keywords
        self.n_presses = n_presses
        self._press_idx = None

    def _find_action_index(self):
        # Works with your Environment.action_space_description coming from Focus.PARSED_ACTIONS
        names = getattr(self.env, "action_space_description", [])
        for i, name in enumerate(names):
            for k in self.keywords:
                if k in name:  # name like "UP", "UPFIRE", etc.
                    return i
        return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._press_idx is None:
            self._press_idx = self._find_action_index()
        if self._press_idx is not None:
            # hold the serve button a few frames to be safe
            for _ in range(self.n_presses):
                obs, _, terminated, truncated, info = self.env.step(self._press_idx)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
        return obs, info




# Load model and environment
focus_dir="resources/focusfiles/"
def load_model_and_env(game, seed, mods, pruned=False):
    if pruned:
        model_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc_pruned/best_model.zip"
        vecnormalize_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc_pruned/best_vecnormalize.pkl"
        focus_file="resources/checkpoints/" + game.capitalize() + "_seed"+str(seed)+"_reward-env_oc_pruned/pruned_"+game.lower()+".yaml"
        print(focus_file)
        env = Environment(env_name="ALE/"+game.capitalize()+"-v5", mods=mods, draw_features=True, reward=0, focus_dir=focus_dir,focus_file=focus_file)
        #env = PressOnReset(env, keywords=("UP", "UPFIRE", "FIRE"), n_presses=5)
    else:
        model_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc/best_model.zip"
        vecnormalize_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc/best_vecnormalize.pkl"
        env = Environment("ALE/"+game.capitalize()+"-v5", draw_features=True, reward=0, focus_dir=focus_dir)

    model = PPO.load(model_path)
    dummy_vecenv = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vecnormalize_path, dummy_vecenv)
    env.training = False
    env.norm_reward = False

    return model, env

def load_model_and_env_lower(game, seed, mods, pruned=False):
    if pruned:
        model_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc_pruned_lower/best_model.zip"
        vecnormalize_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc_pruned_lower/best_vecnormalize.pkl"
        focus_file="resources/checkpoints/" + game.capitalize() + "_seed"+str(seed)+"_reward-env_oc_pruned_lower/pruned_"+game.lower()+".yaml"
        env = Environment(env_name="ALE/"+game.capitalize()+"-v5", mods=mods, draw_features=True, reward=0, focus_dir=focus_dir,focus_file=focus_file)
        #env = PressOnReset(env, keywords=("UP", "UPFIRE", "FIRE"), n_presses=5)
    else:
        model_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc/best_model.zip"
        vecnormalize_path = "resources/checkpoints/"+game.capitalize()+"_seed"+str(seed)+"_reward-env_oc/best_vecnormalize.pkl"
        env = Environment("ALE/"+game.capitalize()+"-v5", draw_features=True, reward=0, focus_dir=focus_dir)

    model = PPO.load(model_path)
    dummy_vecenv = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vecnormalize_path, dummy_vecenv)
    env.training = False
    env.norm_reward = False

    return model, env


# De-normalize observation values
def denormalize_observation(obs, obs_rms, indices):
    mean = obs_rms.mean[indices]
    var = obs_rms.var[indices]
    return obs[indices] * np.sqrt(var) + mean

# Group features by objects with t-1 features being not combined with t ones
def group_features_by_objects_separatet1(feature_names):
    object_groups = defaultdict(list)
    pattern = re.compile(r'^(\w+)\.(x|y)(\[[^\]]+\])?$')

    for feature in feature_names:
        print(feature)
        match = pattern.match(feature)
        if match:
            print(f"Matched: {match.groups()}")
            object_name = match.group(1)  # base name
            property_name = match.group(2)  # x or y
            suffix = match.group(3) or ""  # Optional [t-1]
            group_name = f"{object_name}{suffix}"
            object_groups[group_name].append(feature)
        else:
            print(f"Skipped: {feature}")

    return dict(object_groups)

# Group features by objects with t-1 features being counted as t as well
def group_features_by_objects_combinet1(feature_names):
    object_groups = defaultdict(list)
    pattern = re.compile(r'^(\w+)\.(x|y)(\[[^\]]+\])?$')

    for feature in feature_names:
        match = pattern.match(feature)
        if match:
            object_name = match.group(1)
            group_name = object_name
            object_groups[group_name].append(feature)
        else:
            print(f"Skipped feature: {feature}")

    return dict(object_groups)


def get_vector_entry_descriptions(env):
    # Check if the environment has an attribute 'feature_names'
    if hasattr(env, 'feature_names'):
        return env.feature_names

    # If not, check if the environment has a method to get feature names
    elif hasattr(env, 'get_feature_names'):
        return env.get_feature_names()

    # If neither, attempt to generate generic feature names based on observation space
    else:
        observation_space = env.observation_space
        if hasattr(observation_space, 'names') and observation_space.names:
            return observation_space.names
        elif isinstance(observation_space, gymnasium.spaces.Box):
            # Generate generic feature names for Box spaces
            return [f"feature_{i}" for i in range(observation_space.shape[0])]
        else:
            raise AttributeError("Environment does not provide feature names.")


def group_features_current_only(feature_names):
    EXCLUDE = {"OxygenBar1", "Chicken2"}
    groups = defaultdict(list)
    xy_re  = re.compile(r'^(\w+)\.(x|y)$')
    for f in feature_names:
        m = xy_re.match(f)
        if m:
            obj = f.split('.')[0]
            if obj in EXCLUDE:
                continue
            groups[m.group(1)].append(f)
    return dict(groups)

def group_features_current_and_prev(feature_names):
    tmp = defaultdict(lambda: {"now": {"x": None, "y": None},
                               "prev": {"x": None, "y": None}})

    for name in feature_names:
        if ".x" not in name and ".y" not in name:
            continue                                    # skip non-spatial

        # split "Car1.x[t-1]"  →  "Car1.x", "[t-1]"
        base_part = name.split("[")[0]                  # "Car1.x"
        obj, coord = base_part.split(".", 1)            # "Car1", "x"

        bucket = "prev" if "[t-1]" in name else "now"
        if tmp[obj][bucket][coord] is None:             # keep first occ.
            tmp[obj][bucket][coord] = name

    groups = {}
    for obj, d in tmp.items():
        if d["now"]["x"] and d["now"]["y"]:             # must have current x,y
            groups[obj] = [d["now"]["x"], d["now"]["y"]]
            if d["prev"]["x"] and d["prev"]["y"]:       # add t-1 if present
                groups[obj] += [d["prev"]["x"], d["prev"]["y"]]

    return groups
