from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict
from abc import ABC, abstractmethod
import numpy as np
import random
import re
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm
from attentive_tree_extraction.utils import LogProbQ

RankedSample = Tuple[np.ndarray, int, List[str]]

# ============= OBLIQUE FUNCTIONS =============
def oblique_fn_freeway(att_names):
    axis_re = re.compile(r'\.(x|y)\b')
    axis = []
    for name in att_names:
        m_axis = axis_re.search(name)
        axis.append(m_axis.group(1) if m_axis else None)

    axis = np.asarray(axis)
    n = len(att_names)
    i, j = np.tril_indices(n, k=-1)

    same_axis = axis[i] == axis[j]
    keep = np.nonzero(same_axis)[0]

    return i[keep], j[keep]

def oblique_fn_tennis(att_names):
    _RX = re.compile(r'^(?P<obj>[^.]+)\.(?P<axis>x|y)(?P<lag>\[t-1\])?$')
    meta = []
    for n in att_names:
        m = _RX.match(n)
        if m:
            obj = m.group('obj')
            axis = m.group('axis')
            time = 'prev' if m.group('lag') else 'curr'
            meta.append((obj, axis, time))
        else:
            meta.append((None, None, None))

    ii, jj = [], []
    for i in range(len(att_names)):
        obj_i, axis_i, time_i = meta[i]
        if axis_i is None: continue
        for j in range(i):
            obj_j, axis_j, time_j = meta[j]
            if axis_j is None: continue

            # Rule 1: same axis only
            if axis_i != axis_j:
                continue

            # Rule 2: if any lag must be same object
            if (time_i == 'prev' or time_j == 'prev') and (obj_i != obj_j):
                continue

            ii.append(i); jj.append(j)

    return np.asarray(ii, int), np.asarray(jj, int)

# ============= BASE CLASS =============
class ContextVIPERBase(ABC):
    def __init__(
            self,
            model,
            env,
            feature_names: List[str],
            object_groups: Dict[str, List[str]],
            *,
            reference_object=None,
            distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
            max_depth: int = 5,
            objects_per_context: int = 1,
            dagger_batch: int = 30_000,
            contexts: Dict[str, List[str]],
            oblique = False
    ):
        self.model = model
        self.env = env
        self.feature_names = feature_names
        self.object_groups = {
            obj_name: indices
            for obj_name, indices in object_groups.items()
            if not obj_name.startswith("D(")
        }
        self.ref_obj = reference_object if reference_object is not None else next(iter(object_groups))
        self.distance_fn = distance_fn or (lambda p, q: np.linalg.norm(p - q))
        self.k = objects_per_context
        self.max_depth = max_depth
        self.dagger_batch = dagger_batch
        self.oblique = oblique

        self.logprob_q = LogProbQ(model, env)
        self.adt_by_ctx: dict[str, DecisionTreeClassifier] = {}
        self.dataset: list[RankedSample] = []

        # indices of the coords each object owns
        self._idx = {o: [feature_names.index(f) for f in feats]
                     for o, feats in object_groups.items()}

        self.contexts = {k: set(v) for k, v in contexts.items()}

        # Game-specific initialization
        self._init_game_specific()

        print("INDEX CHECK")
        print("player :", self._idx[self.ref_obj])
        for o in list(self.object_groups)[:len(self.object_groups)]:
            if o != self.ref_obj:
                print(o, self._idx[o])

    @abstractmethod
    def _init_game_specific(self):
        """Initialize game-specific attributes and names"""
        pass

    @abstractmethod
    def _attention_objects(self, state, ctx_key):
        """Return the k nearest non-reference objects for the context"""
        pass

    @abstractmethod
    def _att_vec(self, state, ctx_key):
        """Build attention vector for the state and context"""
        pass

    @abstractmethod
    def _ctx_key(self, state):
        """Determine context key from state or ranking"""
        pass

    @abstractmethod
    def _collect(self, n):
        """Collect n samples from teacher"""
        pass

    def _coords(self, state: np.ndarray, obj):
        idx = self._idx[obj][:2]
        return state[idx]

    def _coords_prev(self, state: np.ndarray, obj: str):
        """x(t-1), y(t-1)"""
        return state[self._idx[obj][2:4]]

    def _distance_ranking(self, state):
        ref = self._coords(state, self.ref_obj)
        rank = []
        for obj in self.object_groups:
            if obj == self.ref_obj:
                continue
            pos = self._coords(state, obj)
            dist = np.inf if np.isnan(pos).any() else self.distance_fn(ref, pos)
            rank.append((obj, dist))
        rank.sort(key=lambda kv: kv[1])
        return [o for o, _ in rank]

    def _teacher_step(self, obs):
        act, _ = self.model.predict(obs, deterministic=True)
        return int(act[0])

    def _predict_with_tree(self, tree, att_vec_1d: np.ndarray) -> int:
        z = att_vec_1d
        if hasattr(self, "_keep_idx"):
            z = z[self._keep_idx]
        if getattr(tree, "_is_oblique", False):
            z = np.hstack([z, z[tree._oblique_i] - z[tree._oblique_j]])
        return int(tree.predict(z.reshape(1, -1))[0])

    def _train_adt_once(self, data_by_ctx):
        """Train one CART per context on attention vectors."""
        self.adt_by_ctx.clear()
        for ctx, d in tqdm(data_by_ctx.items(), desc="Training ADTs"):
            X = np.asarray(d["att"])
            y = np.asarray(d["actions"])

            # Get VIPER weights
            viper_weights = np.array([self.logprob_q.get_disagreement_cost(s).item()
                                      for s in d["states"]])
            w = viper_weights

            # Build oblique features if enabled
            if self.oblique and len(self.oblique_i) > 0:
                X_tr = np.hstack([X, X[:, self.oblique_i] - X[:, self.oblique_j]])
                oblique_feature_names = [
                    f"{self.att_names[i]} - {self.att_names[j]}"
                    for i, j in zip(self.oblique_i, self.oblique_j)
                ]
                feature_names_all = self.att_names + oblique_feature_names
            else:
                X_tr = X
                feature_names_all = list(self.att_names)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=5,
                random_state=0,
                class_weight="balanced"
            )
            tree.fit(X_tr, y, sample_weight=w)

            # Store metadata
            tree._is_oblique = bool(self.oblique and len(self.oblique_i) > 0)
            if tree._is_oblique:
                tree._oblique_i = self.oblique_i
                tree._oblique_j = self.oblique_j

            # Diagnostics
            train_acc = tree.score(X_tr, y)
            print(f"[{ctx}] train accuracy = {train_acc:.1%}")
            print(f"[{ctx}] Action distribution: {np.bincount(y)}")

            self.adt_by_ctx[ctx] = tree

    # DAGGER refinement loop
    def refine(self, iterations: int = 3):
        # Initial teacher data
        self.dataset.extend(self._collect(self.dagger_batch))

        best_score = -float('inf')
        best_trees = {}

        for it in tqdm(range(iterations), desc="Refinement"):
            # Bucket by context
            buckets = defaultdict(lambda: {"states": [], "att": [], "actions": []})
            for s, a, r in self.dataset:
                key = self._ctx_key(s) if hasattr(self, '_ctx_key_uses_state') else self._ctx_key(r)
                buckets[key]["states"].append(s)
                buckets[key]["actions"].append(a)
                att = self._att_vec(s, key)
                buckets[key]["att"].append(att)

            for k, v in buckets.items():
                print(k, len(v["att"]))

            # Train trees
            self._train_adt_once(buckets)

            # Evaluate
            _, ep_returns = self.evaluate(num_episodes=3, verbose=False)
            current_score = np.mean(ep_returns)
            print(f"[iter {it}] Score: {current_score:.1f}")

            if current_score > best_score:
                best_score = current_score
                best_trees = self.adt_by_ctx.copy()
                print(f"[iter {it}] NEW BEST: {best_score:.1f}")

            # DAGGER: run student, query teacher
            obs = self.env.reset()
            new_samples = []

            pbar = tqdm(desc=f"DAGGER iter {it}", total=self.dagger_batch, unit="samples")

            for _ in range(self.dagger_batch):
                ranking = self._distance_ranking(obs[0])
                key = self._ctx_key(obs[0]) if hasattr(self, '_ctx_key_uses_state') else self._ctx_key(ranking)
                adt = self.adt_by_ctx.get(key)

                if adt is not None:
                    att_vec = self._att_vec(obs[0], key)
                    env_act = self._predict_with_tree(adt, att_vec)
                else:
                    env_act = self._teacher_step(obs)

                teacher_act = self._teacher_step(obs)
                new_samples.append((obs[0].copy(), teacher_act, ranking))

                obs, _, done, _ = self.env.step(np.array([env_act], np.int64))
                if done:
                    obs = self.env.reset()

                pbar.update(1)

            pbar.close()
            self.dataset.extend(new_samples)
            print(f"[iter {it}] dataset size → {len(self.dataset):,}")

        if best_trees:
            self.adt_by_ctx = best_trees
            print(f"Restored best trees with score: {best_score:.1f}")

    def evaluate(self, num_episodes: int = 10, verbose: bool = False):
        report, ep_returns = [], []
        steps_with_tree = 0
        ctx_stats = defaultdict(lambda: {"steps": 0, "matches": 0})

        obs = self.env.reset()
        epret, episode_idx = 0.0, 0
        pbar = tqdm(total=num_episodes, desc="Evaluating") if not verbose else None

        while episode_idx < num_episodes:
            teacher_act = self._teacher_step(obs)
            ranking = self._distance_ranking(obs[0])
            ctx_key = self._ctx_key(obs[0]) if hasattr(self, '_ctx_key_uses_state') else self._ctx_key(ranking)
            adt = self.adt_by_ctx.get(ctx_key)

            if adt is not None:
                att_vec = self._att_vec(obs[0], ctx_key)
                env_act = self._predict_with_tree(adt, att_vec)
                steps_with_tree += 1
                ctx_stats[ctx_key]["steps"] += 1
                ctx_stats[ctx_key]["matches"] += int(env_act == teacher_act)
            else:
                env_act = teacher_act

            obs, reward, done, _ = self.env.step(np.array([env_act], np.int64))
            r = float(reward)
            epret += r

            if verbose:
                print(f"[ep{episode_idx:02d}] ctx={ctx_key or '<none>':<20} "
                      f"use_tree={adt is not None} env_act={env_act} "
                      f"teacher_act={teacher_act} R={r:.2f}")

            if done:
                ep_returns.append(epret)
                episode_idx += 1
                if pbar: pbar.update(1)
                epret = 0.0
                obs = self.env.reset()

        if pbar: pbar.close()

        total_steps = sum(s["steps"] for s in ctx_stats.values()) + (len(ep_returns) - 1)
        coverage = steps_with_tree / total_steps if total_steps else 0.0
        mean_step = (sum(ep_returns) / total_steps) if total_steps else float("nan")
        mean_ep = float(np.mean(ep_returns))
        std_ep = float(np.std(ep_returns))

        report.extend([
            f"Episodes rolled out   : {num_episodes}",
            f"Steps (total)         : {total_steps}",
            f"Avg reward / step     : {mean_step:.4f}",
            f"Avg episode return    : {mean_ep:.2f} ± {std_ep:.2f}",
            f"Tree coverage         : {coverage:5.1%}",
            "",
            "Context fidelity (exact matches):",
        ])
        for ctx, s in ctx_stats.items():
            acc = s["matches"] / s["steps"] if s["steps"] else 0.0
            report.append(f"  {ctx:<30} steps={s['steps']:5d} acc={acc:5.1%}")

        for line in report:
            print(line)
        return report, ep_returns

    def save_trees(
            self,
            gamename: str,
            seed: int,
            evaluation: List[str],
            returns,
            obj_per_ctx: int,
            pruned: str = "",
            output_path: str = "context_adt_nearest/",
    ):
        objstr = f"{obj_per_ctx}obj"
        dirname = f"{gamename}_seed{seed}{pruned}{objstr}"
        parent_dir = Path(output_path) / dirname / str(self.max_depth)

        parent_dir.mkdir(parents=True, exist_ok=True)

        existing_nums = []
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                existing_nums.append(int(item.name))

        next_num = 1 if not existing_nums else max(existing_nums) + 1
        itnumber = str(next_num)
        out_dir = parent_dir / itnumber
        out_dir.mkdir(parents=True, exist_ok=True)

        for ctx, tree in self.adt_by_ctx.items():
            path = out_dir / f"adt_{ctx}.joblib"
            dump(tree, path)
            print(f"Saved tree for context {ctx} → {path}")

        summary_file = out_dir / "1evaluation_summary.txt"
        summary_file.write_text("\n".join(evaluation) +
                                "\n" + "rewards: " + str(returns) +
                                "\n" + "context splits" + str(self.contexts))
        print(f"Saved evaluation summary → {summary_file}")
        print(f"All files saved in folder #{next_num}")


# ============= FREEWAY IMPLEMENTATION =============
class FreewayContextVIPER(ContextVIPERBase):
    def _init_game_specific(self):
        self.att_names = (
            ["Player.x", "Player.y", "Player.x[t-1]", "Player.y[t-1]"] +
            [f"Car{i}.{coord}"
             for i in range(self.k)
             for coord in ["dx", "dy", "dx[t-1]", "dy[t-1]"]]
        )
        self.oblique_i, self.oblique_j = oblique_fn_freeway(self.att_names)

    def _attention_objects(self, state, ctx_key):
        """Freeway version: vertical gap based"""
        py = self._coords(state, self.ref_obj)[1]

        def vert_gap(o):
            cy = self._coords(state, o)[1]
            if ctx_key == "top":
                return cy - py if cy > py else np.inf
            if ctx_key == "bottom":
                return py - cy if cy < py else np.inf
            return abs(cy - py)

        candidates = [
            o for o in self.object_groups
            if o != self.ref_obj and (
                ctx_key == "" or o in self.contexts.get(ctx_key, ())
            )
        ]

        if not candidates:
            print(f"[warn] ctx={ctx_key} objects_per_context={self.k} "
                  f"but found 0 candidates.")

        candidates.sort(key=vert_gap)
        return candidates[:self.k]

    # Leveraging relative positions
    def _att_vec(self, state, ctx_key):
        ref_curr = self._coords(state, self.ref_obj)
        ref_prev = self._coords_prev(state, self.ref_obj)

        # Keep player absolute position
        vec = list(ref_curr) + list(ref_prev)

        # Use RELATIVE positions for other objects
        for o in self._attention_objects(state, ctx_key):
            obj_curr = self._coords(state, o)
            obj_prev = self._coords_prev(state, o)
            vec.extend(obj_curr - ref_curr)  # Relative current
            vec.extend(obj_prev - ref_prev)  # Relative previous

        # Pad if needed
        expected_length = 4 * (1 + self.k)
        while len(vec) < expected_length:
            vec.append(0.0)

        return np.asarray(vec, dtype=np.float32)

    # determine context based on nearest object
    def _ctx_key(self, ranking):
        if not ranking:
            return ""
        nearest = ranking[0]
        for ctx_name, obj_set in self.contexts.items():
            if nearest in obj_set:
                return ctx_name
        return ""

    # collect samples via balancing method
    def _collect(self, n):
        n_actions = int(self.env.action_space.n)
        samples_per_action = max(1, n // n_actions)

        action_buffers = {a: [] for a in range(n_actions)}
        obs = self.env.reset()

        pbar = tqdm(desc="Collecting balanced data", unit="samples")
        while any(len(buf) < samples_per_action for buf in action_buffers.values()):
            act = self._teacher_step(obs)
            ranking = self._distance_ranking(obs[0])

            if len(action_buffers[act]) < samples_per_action:
                action_buffers[act].append((obs[0].copy(), act, ranking))
                pbar.update(1)
                pbar.set_postfix({f"act_{i}": len(buf) for i, buf in action_buffers.items()})

            obs, _, done, _ = self.env.step(np.array([act], np.int64))
            if done:
                obs = self.env.reset()

        pbar.close()

        buf = []
        for action_samples in action_buffers.values():
            buf.extend(action_samples[:samples_per_action])

        random.shuffle(buf)

        print(f"Collected balanced dataset: {len(buf)} samples total")
        for act in range(n_actions):
            count = sum(1 for _, a, _ in buf if a == act)
            print(f"  Action {act}: {count} ({count/len(buf)*100:.1f}%)")

        return buf


# ============= TENNIS IMPLEMENTATION =============
class TennisContextVIPER(ContextVIPERBase):
    def _init_game_specific(self):
        self._last_ctx = None
        self._ctx_key_uses_state = True  # Flag to indicate ctx_key uses state not ranking

        # Build attribute names from actual objects
        self.att_objects = sorted([o for o in self.object_groups if o != self.ref_obj])
        self._att_idx = (
            self._idx[self.ref_obj][:4] +
            [i for o in self.att_objects for i in self._idx[o][:4]]
        )
        self.att_names = [self.feature_names[i] for i in self._att_idx]
        self.oblique_i, self.oblique_j = oblique_fn_tennis(self.att_names)

    # Unnormalize observations if using VecNormalize
    def _unnorm_obs(self, s: np.ndarray) -> np.ndarray:
        try:
            rms = self.env.obs_rms
            eps = 1e-8
            std = np.sqrt(rms.var + eps)
            return s * std + rms.mean
        except Exception:
            return s

    # priority based selection
    def _attention_objects(self, state, ctx_key):
        candidates = []

        if ctx_key == "top":
            priority_order = ["Ball1", "Enemy1", "BallShadow1"]
        else:  # bottom
            priority_order = ["Ball1", "Enemy1", "BallShadow1"]

        for obj in priority_order:
            if obj in self.object_groups and obj != self.ref_obj:
                candidates.append(obj)
                if len(candidates) >= self.k:
                    break

        return candidates[:self.k]

    # denormalized version
    def _att_vec(self, state, ctx_key):
        s = self._unnorm_obs(state)
        return s[self._att_idx].astype(np.float32)

    # based on positioning context decision
    def _ctx_key(self, state):
        s = self._unnorm_obs(state)
        p_y = s[self._idx["Player1"][1]]
        e_y = s[self._idx["Enemy1"][1]]

        # Sticky decision to avoid flicker
        margin = 1e-3
        if abs(p_y - e_y) <= margin and self._last_ctx is not None:
            return self._last_ctx

        ctx = "top" if p_y < e_y else "bottom"
        self._last_ctx = ctx
        return ctx

    def _collect(self, n):
        buf = []
        obs = self.env.reset()

        pbar = tqdm(desc="Collecting teacher data", total=n, unit="samples")

        for _ in range(n):
            act = self._teacher_step(obs)
            ranking = self._distance_ranking(obs[0])
            buf.append((obs[0].copy(), act, ranking))

            obs, _, done, _ = self.env.step(np.array([act], np.int64))
            if done:
                obs = self.env.reset()

            pbar.update(1)

        pbar.close()

        action_counts = {}
        for _, a, _ in buf:
            action_counts[a] = action_counts.get(a, 0) + 1

        print(f"Collected {len(buf)} samples with distribution: {action_counts}")
        return buf


# ============= FACTORY FUNCTION =============
def ContextVIPER(game: str, **kwargs):
    """
    Factory function to create the appropriate ContextVIPER instance

    Args:
        game: Either "freeway" or "tennis"
        **kwargs: All other arguments passed to the constructor

    Returns:
        FreewayContextVIPER or TennisContextVIPER instance
    """
    game = game.lower()
    if game == "freeway":
        return FreewayContextVIPER(**kwargs)
    elif game == "tennis":
        return TennisContextVIPER(**kwargs)
    else:
        raise ValueError(f"Unknown game: {game}. Supported games are 'freeway' and 'tennis'")