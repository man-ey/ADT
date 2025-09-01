import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
from attentive_tree_extraction.adt_extractor import ContextVIPER
import utils.general_utils as general_utils

# Default contexts for each game
GAME_CONTEXTS = {
    "freeway": {
        "top": ["Car6", "Car7", "Car8", "Car9", "Car10"],
        "bottom": ["Car1", "Car2", "Car3", "Car4", "Car5"]
    },
    "tennis": {
        "top": ["Ball1", "Enemy1", "BallShadow1"],
        "bottom": ["Ball1", "Enemy1", "BallShadow1"]
    }
}

# Default distance functions for each game
DISTANCE_FUNCTIONS = {
    "freeway": lambda p, q: np.linalg.norm(p - q, ord=2),
    "tennis": lambda p, q: np.linalg.norm(p - q, ord=2)
}


def train_trees(args):
    # Setup game name
    game_name = args.game.capitalize()

    # Load model and environment
    print(f"Loading {game_name} environment with seed {args.seed}...")
    model, env = general_utils.load_model_and_env(
        game_name,
        args.seed,
        mods=None,
        pruned=args.pruned
    )

    # Get feature names and object groups
    obs = env.reset()
    scobi_env = env.venv.envs[0]
    feature_names = scobi_env.get_vector_entry_descriptions()
    object_groups = general_utils.group_features_current_and_prev(feature_names)

    if args.game == "freeway":
        # Remove unplayable player character
        object_groups.pop("Chicken2", None)

    # Get contexts (use custom if provided, otherwise defaults)
    contexts = GAME_CONTEXTS[args.game]

    # Select distance function
    if args.distance == "default":
        distance_fn = DISTANCE_FUNCTIONS[args.game]
    elif args.distance == "l2":
        distance_fn = lambda p, q: np.linalg.norm(p - q, ord=2)
    elif args.distance == "l1":
        distance_fn = lambda p, q: np.linalg.norm(p - q, ord=1)
    elif args.distance == "weighted":
        distance_fn = lambda p, q: np.sqrt((p[0] - q[0])**2 + 0.3 * (p[1] - q[1])**2)
    else:
        distance_fn = DISTANCE_FUNCTIONS[args.game]

    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Game:                {game_name}")
    print(f"Seed:                {args.seed}")
    print(f"Pruned:              {args.pruned}")
    print(f"Max depth:           {args.depth}")
    print(f"Objects per context: {args.k}")
    print(f"DAGGER batch size:   {args.batch}")
    print(f"DAGGER iterations:   {args.iterations}")
    print(f"Oblique features:    {args.oblique}")
    print(f"Distance function:   {args.distance}")
    print(f"Output path:         {args.output}")
    print(f"\nContexts:")
    for ctx_name, objects in contexts.items():
        print(f"  {ctx_name}: {objects}")
    print(f"\nObject groups ({len(object_groups)}):")
    for obj_name in list(object_groups.keys())[:10]:
        print(f"  - {obj_name}")
    if len(object_groups) > 10:
        print(f"  ... and {len(object_groups) - 10} more")
    print("="*60 + "\n")

    # Create extractor
    print("Initializing ContextVIPER...")
    extractor = ContextVIPER(
        game=args.game,
        model=model,
        env=env,
        feature_names=feature_names,
        object_groups=object_groups,
        max_depth=args.depth,
        objects_per_context=args.k,
        dagger_batch=args.batch,
        contexts=contexts,
        distance_fn=distance_fn,
        oblique=args.oblique
    )

    # Train with DAGGER
    print(f"\nStarting DAGGER refinement ({args.iterations} iterations)...")
    extractor.refine(iterations=args.iterations)

    # Evaluate
    print(f"\nEvaluating on {args.eval_episodes} episodes...")
    report, ep_returns = extractor.evaluate(num_episodes=args.eval_episodes)

    print("\nEpisode returns:", ep_returns)
    print(f"Mean return: {np.mean(ep_returns):.2f} ± {np.std(ep_returns):.2f}")

    # Save trees
    if args.save:
        print("\nSaving trees...")
        pruned_str = "_pruned_" if args.pruned else "_"
        extractor.save_trees(
            gamename=game_name,
            seed=args.seed,
            evaluation=report,
            returns=ep_returns,
            obj_per_ctx=args.k,
            pruned=pruned_str,
            output_path=args.output
        )
        print(f"Trees saved to {args.output}")

    return extractor, ep_returns


def main():
    parser = argparse.ArgumentParser(
        description="Train context-based decision trees",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("-game", type=str, required=True,
                        choices=["freeway", "tennis"],
                        help="Which game to train on")

    # Environment settings
    parser.add_argument("-seed", type=int, default=0,
                        help="Random seed for environment")
    parser.add_argument("-pruned", action="store_true",
                        help="Use pruned environment")

    # Tree settings
    parser.add_argument("-depth", type=int, default=5,
                        help="Maximum depth of decision trees")
    parser.add_argument("-k", type=int, default=3,
                        help="Number of objects per context")
    parser.add_argument("-oblique", action="store_true",
                        help="Use oblique decision boundaries")

    # Training settings
    parser.add_argument("-batch", type=int, default=30000,
                        help="DAGGER batch size (samples per iteration)")
    parser.add_argument("-iterations", type=int, default=3,
                        help="Number of DAGGER iterations")

    # Distance function
    parser.add_argument("-distance", type=str, default="default",
                        choices=["default", "l2", "l1", "weighted"],
                        help="Distance function for attention selection")

    # Evaluation
    parser.add_argument("-eval-episodes", type=int, default=10,
                        help="Number of episodes for evaluation")

    # Output
    parser.add_argument("-output", type=str, default="context_adt_nearest/",
                        help="Output directory for saved trees")
    parser.add_argument("-no-save", dest="save", action="store_false",
                        help="Don't save trees after training")

    args = parser.parse_args()

    # Run training
    try:
        extractor, returns = train_trees(args)

        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Final mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        if args.save:
            print(f"Trees saved to: {args.output}")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
