import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import imageio

# Import the extractor classes
from attentive_tree_extraction.adt_extractor import ContextVIPER, FreewayContextVIPER, TennisContextVIPER

# Wrapper loading trees
class TreeLoader:
    def __init__(self, viper_instance, tree_dir: str):
        self.viper = viper_instance
        self._tree_dir = Path(tree_dir)

        # Load trees into the viper instance
        self._load_trees()

    def _load_trees(self):
        # Load trees from directory into the instance
        tree_paths = list(self._tree_dir.glob("adt_*.joblib"))
        print(f"Looking for trees in: {self._tree_dir}")
        print(f"Found {len(tree_paths)} tree files")

        for p in tree_paths:
            ctx = p.stem[len("adt_"):]
            tree = load(p)
            self.viper.adt_by_ctx[ctx] = tree
            print(f"Loaded tree for context: '{ctx}'")

            if hasattr(tree, '_is_oblique') and tree._is_oblique:
                print(f"  - Tree is oblique with {len(tree._oblique_i)} pairs")

            n_features = tree.n_features_in_ if hasattr(tree, 'n_features_in_') else tree.n_features_
            print(f"  - Tree expects {n_features} features")

    def predict_with_tree(self, state: np.ndarray):
        # Determine context
        if isinstance(self.viper, TennisContextVIPER):
            ctx_key = self.viper._ctx_key(state)
        else:  # FreewayContextVIPER
            ranking = self.viper._distance_ranking(state)
            ctx_key = self.viper._ctx_key(ranking)

        adt = self.viper.adt_by_ctx.get(ctx_key)

        if adt is None:
            return None, ctx_key

        # Get attention vector
        att_vec = self.viper._att_vec(state, ctx_key)

        # Use tree predict method
        action = self.viper._predict_with_tree(adt, att_vec)
        return action, ctx_key

    def evaluate(self, num_episodes: int = 10, verbose: bool = False):
        ep_returns = []
        ctx_stats = defaultdict(lambda: {"steps": 0, "matches": 0})

        obs = self.viper.env.reset()
        epret = 0.0
        episode_idx = 0
        pbar = tqdm(total=num_episodes, desc="Evaluating")

        while episode_idx < num_episodes:
            teacher_act = self.viper._teacher_step(obs)
            env_act, ctx_key = self.predict_with_tree(obs[0])

            if env_act is not None:
                ctx_stats[ctx_key]["steps"] += 1
                ctx_stats[ctx_key]["matches"] += int(env_act == teacher_act)
            else:
                env_act = teacher_act

            obs, reward, done, _ = self.viper.env.step(np.array([env_act], np.int64))
            epret += float(reward)

            if verbose:
                print(f"ctx={ctx_key:6} tree_act={env_act} teacher_act={teacher_act} R={reward:.2f}")

            if done:
                ep_returns.append(epret)
                episode_idx += 1
                pbar.update(1)
                epret = 0.0
                obs = self.viper.env.reset()

        pbar.close()

        # Build report
        mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
        std_ret = float(np.std(ep_returns)) if ep_returns else float("nan")

        lines = []
        lines.append("=== Tree Evaluation ===")
        lines.append(f"Episodes rolled out   : {num_episodes}")
        lines.append(f"Avg episode return    : {mean_ret:.2f} ± {std_ret:.2f}")
        lines.append("")
        lines.append("Context fidelity (exact matches):")

        total_steps = sum(s["steps"] for s in ctx_stats.values())
        for ctx, stats in ctx_stats.items():
            if stats["steps"] > 0:
                acc = stats["matches"] / stats["steps"]
                lines.append(f"  {ctx:<30} steps={stats['steps']:5d} acc={acc:5.1%}")

        if total_steps > 0:
            coverage = sum(s["steps"] for s in ctx_stats.values()) / (total_steps + episode_idx)
            lines.append(f"\nTree coverage: {coverage:5.1%}")

        # Save summary
        try:
            out_txt = self._tree_dir / "evaluation_summary.txt"
            body = "\n".join(lines)
            body += "\n\nRewards per episode: " + str(ep_returns)
            body += "\n\nContexts: " + str(self.viper.contexts)
            out_txt.write_text(body)
            print(f"\nSaved evaluation summary → {out_txt}")
        except Exception as e:
            print(f"Failed to write evaluation file: {e}")

        print("\n".join(lines))
        return ep_returns

    # Create video of tree performance
    def visualize(self, num_episodes: int = 1, video_path: str = None):
        if video_path is None:
            video_path = self._tree_dir / "evaluation_video.mp4"

        obs = self.viper.env.reset()
        frames = []
        episode_idx = 0
        fig, ax = plt.subplots()
        img = ax.imshow(self.viper.env.venv.envs[0].obj_obs)
        plt.axis("off")
        pbar = tqdm(total=num_episodes, desc="Recording video")

        while episode_idx < num_episodes:
            teacher_act = self.viper._teacher_step(obs)
            env_act, ctx_key = self.predict_with_tree(obs[0])

            if env_act is None:
                env_act = teacher_act

            obs, reward, done, _ = self.viper.env.step(np.array([env_act], np.int64))

            frame = self.viper.env.venv.envs[0].obj_obs.copy()
            ax.set_title(f"ctx={ctx_key} act={env_act} teacher={teacher_act}")
            img.set_data(frame)
            plt.pause(1.0 / 30)
            frames.append(frame.copy())

            if done:
                pbar.update(1)
                episode_idx += 1
                obs = self.viper.env.reset()

        pbar.close()
        plt.close()

        with imageio.get_writer(str(video_path), fps=30) as writer:
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (255 * (frame / frame.max())).astype(np.uint8)
                writer.append_data(frame)
        print(f"Saved video to {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Load and evaluate decision trees")
    parser.add_argument("-dir", "--directory", type=str, required=True,
                        help="Path to directory containing tree files (adt_*.joblib)")
    parser.add_argument("-game", type=str, required=True, choices=["freeway", "tennis"],
                        help="Which game to load (freeway or tennis)")
    parser.add_argument("-episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("-video", action="store_true",
                        help="Generate visualization video")
    parser.add_argument("-seed", type=int, default=0,
                        help="Seed for the game environment")
    parser.add_argument("-pruned", action="store_true",
                        help="Use pruned environment")
    parser.add_argument("-verbose", action="store_true",
                        help="Verbose output during evaluation")
    parser.add_argument("-k", type=int, default=3,
                        help="Objects per context (default: 3)")

    args = parser.parse_args()

    # Load model and environment
    from utils.general_utils import load_model_and_env, group_features_current_and_prev

    game_name = args.game.capitalize()
    model, env = load_model_and_env(game_name, args.seed, None, pruned=args.pruned)

    obs = env.reset()
    scobi_env = env.venv.envs[0]
    feature_names = scobi_env.get_vector_entry_descriptions()
    object_groups = group_features_current_and_prev(feature_names)

    # Remove unplayable char for Freeway
    if game_name == "Freeway":
        object_groups.pop("Chicken2", None)

    # Define contexts based on game
    if args.game == "freeway":
        contexts = {}
        tree_dir = Path(args.directory)
        for tree_file in tree_dir.glob("adt_*.joblib"):
            ctx_name = tree_file.stem[len("adt_"):]
            if ctx_name and ctx_name != "":
                # Try to infer objects from context name or use defaults
                contexts[ctx_name] = set()

        # If no specific contexts found, use defaults
        if not contexts or len(contexts) == 0:
            contexts = {
                "top": {f"Car{i}" for i in range(6, 11)},
                "bottom": {f"Car{i}" for i in range(1, 6)}
            }
    else:  # tennis
        contexts = {
            "top": ["Ball1", "Enemy1", "BallShadow1"],
            "bottom": ["Ball1", "Enemy1", "BallShadow1"]
        }

    # Create ContextVIPER instance
    viper = ContextVIPER(
        game=args.game,
        model=model,
        env=env,
        feature_names=feature_names,
        object_groups=object_groups,
        contexts=contexts,
        objects_per_context=args.k,
        oblique=True,
        dagger_batch=0  # not training, just loading
    )

    # Create loader and load trees into viper
    loader = TreeLoader(viper, args.directory)

    # Run evaluation
    print(f"\nEvaluating {game_name} trees from: {args.directory}")
    print(f"Running {args.episodes} episodes...")
    loader.visualize(num_episodes=min(3, args.episodes))

    returns = loader.evaluate(num_episodes=args.episodes, verbose=args.verbose)

    # Generate video if requested
    if args.video:
        print(f"\nGenerating video...")
        loader.visualize(num_episodes=min(3, args.episodes))


if __name__ == "__main__":
    main()