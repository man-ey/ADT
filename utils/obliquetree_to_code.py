#!/usr/bin/env python3
"""
obliquetree_to_code.py - Updated to work with both Freeway and Tennis
"""
from __future__ import annotations

import glob
import math
import os
import re
import textwrap
from pathlib import Path

import joblib
import numpy as np

#  helper functions
def get_feature_importance_info(tree, feature_names):
    importances = tree.feature_importances_
    used_features = []
    for i, (name, imp) in enumerate(zip(feature_names, importances)):
        if imp > 0:
            used_features.append((i, name, imp))
    return used_features

def get_tree_feature_usage(tree):
    used_features = set()

    def traverse(node=0):
        if tree.tree_.feature[node] != -2:  # Not a leaf
            used_features.add(tree.tree_.feature[node])
            traverse(tree.tree_.children_left[node])
            traverse(tree.tree_.children_right[node])

    traverse()
    return sorted(used_features)

# Recursively turn an sklearn tree into nested if/else code
def extract_tree_body_as_code(tree, feature_names, class_names,
                              node=0, depth=1) -> str:
    if tree.tree_.feature[node] == -2:  # leaf
        pred = int(class_names[tree.tree_.value[node].argmax()])
        return f"{'    '*depth}return {pred}\n"

    i      = int(tree.tree_.feature[node])

    # Safety check for feature index
    if i >= len(feature_names):
        print(f"WARNING: Feature index {i} out of range (max {len(feature_names)-1})")
        fname = f"feature_{i}"
    else:
        fname = feature_names[i]

    thresh = float(tree.tree_.threshold[node])

    cond = f"input_features['{fname}'] <= {thresh:.6f}"

    left  = extract_tree_body_as_code(tree, feature_names, class_names,
                                      tree.tree_.children_left[node], depth+1)
    right = extract_tree_body_as_code(tree, feature_names, class_names,
                                      tree.tree_.children_right[node], depth+1)

    return (
        f"{'    '*depth}if {cond}:\n"
        f"{left}"
        f"{'    '*depth}else:\n"
        f"{right}"
    )

def detect_game_from_path(path):
    path_str = str(path).lower()
    if 'freeway' in path_str:
        return 'freeway'
    elif 'tennis' in path_str:
        return 'tennis'
    else:
        return None

def reconstruct_freeway_features(k, context_name=""):
    # Base features for the reference object (Chicken)
    base_features = [
        "Chicken1.X",      # ref.x
        "Chicken1.Y",      # ref.y
        "Chicken1.X[t-1]", # ref.x[t-1]
        "Chicken1.Y[t-1]", # ref.y[t-1]
    ]

    for i in range(k):
        # relative positions (obj_pos - chicken_pos)
        base_features.extend([
            f"Car{i+1}.X-Chicken1.X",      # obj{i}_rel.dx
            f"Car{i+1}.Y-Chicken1.Y",      # obj{i}_rel.dy
            f"Car{i+1}.X[t-1]-Chicken1.X[t-1]", # obj{i}_rel.dx[t-1]
            f"Car{i+1}.Y[t-1]-Chicken1.Y[t-1]", # obj{i}_rel.dy[t-1]
        ])

    return base_features

def reconstruct_tennis_features(k, context_name=""):
    # The base implementation uses absolute positions (unnormalized)
    base_features = [
        "Player1.x",
        "Player1.y",
        "Player1.x[t-1]",
        "Player1.y[t-1]",
    ]

    # Add other objects based on priority order
    priority_objects = ["Ball1", "Enemy1", "BallShadow1"]
    for i in range(min(k, len(priority_objects))):
        obj = priority_objects[i]
        base_features.extend([
            f"{obj}.x",
            f"{obj}.y",
            f"{obj}.x[t-1]",
            f"{obj}.y[t-1]",
        ])

    return base_features

def reconstruct_feature_names(k, context_name="", game="freeway"):
    if game == 'freeway':
        return reconstruct_freeway_features(k, context_name)
    elif game == 'tennis':
        return reconstruct_tennis_features(k, context_name)
    else:
        raise ValueError(f"Unknown game: {game}")

def generate_oblique_features(base_names, oblique_i, oblique_j):
    oblique_names = []
    for idx in range(len(oblique_i)):
        i_idx = oblique_i[idx]
        j_idx = oblique_j[idx]
        if i_idx < len(base_names) and j_idx < len(base_names):
            feat_i = base_names[i_idx]
            feat_j = base_names[j_idx]
            oblique_names.append(f"({feat_i})-({feat_j})")
        else:
            oblique_names.append(f"oblique_{idx}")
    return oblique_names

def build_final_script(env_name, vecnorm_path, focus_file_path, seed,
                       tree_body_code, feature_names, base_feature_count,
                       oblique_i, oblique_j, is_oblique):

    # Generate oblique transformation code if needed
    if is_oblique and len(oblique_i) > 0:
        oblique_transform = f"""        # Apply oblique transformation
        base_features = vec[:{base_feature_count}]
            oblique_i = {oblique_i.tolist()}
            oblique_j = {oblique_j.tolist()}
oblique_features = []
for i, j in zip(oblique_i, oblique_j):
if i < len(base_features) and j < len(base_features):
oblique_features.append(base_features[i] - base_features[j])
vec = list(base_features) + oblique_features
            """
    else:
        oblique_transform = "        "

    raw = f'''\
FEATURE_NAMES = {feature_names!r}
IS_OBLIQUE = {is_oblique}

ENV_NAME      = "{env_name}"
SEED          = {seed}

def if_checks(input_features):
    vec = input_features[0]
    input_features = {{k: v for k, v in zip(FEATURE_NAMES, vec)}}
{tree_body_code.rstrip()}
'''
    return textwrap.dedent(raw)


def main() -> None:
    # process multiple games/folders by updating this list
    folders_to_process = [
        {
            'path': "context_adt_nearest/Tennis_seed0_pruned_3obj/tennis/",
            'game': 'tennis',
            'seed': 0,
            'env_name': 'ALE/Tennis-v5',
            'focus_file': 'pruned_tennis.yaml',
            'vecnorm_path': 'best_vecnormalize.pkl'
        },
    ]

    for config in folders_to_process:
        base_folder_path = config['path']
        game = config['game']

        for tree_path in glob.glob(os.path.join(base_folder_path, "**/adt_*.joblib"),
                                   recursive=True):
            print("\n" + "="*60)
            print(f"Processing {tree_path}")
            print(f"Game: {game}")
            print("="*60)

            tree = joblib.load(tree_path)

            # Extract context name from filename
            tree_filename = Path(tree_path).stem
            context_name = tree_filename.replace("adt_", "")

            # Check if tree has stored feature names
            if hasattr(tree, '_base_feature_names'):
                # Tree has stored feature names
                base_feature_names = tree._base_feature_names
                if hasattr(tree, '_feature_names_all'):
                    feature_names = tree._feature_names_all
                else:
                    feature_names = base_feature_names
                base_feature_count = len(base_feature_names)
                k = (base_feature_count // 4) - 1 if base_feature_count > 4 else 1

                print(f"Using stored feature names from tree")
                print(f"  - Base features: {len(base_feature_names)}")

            else:
                # Check if tree has oblique attributes
                is_oblique = hasattr(tree, '_is_oblique') and tree._is_oblique

                if is_oblique and hasattr(tree, '_oblique_i'):
                    oblique_i = np.array(tree._oblique_i)
                    oblique_j = np.array(tree._oblique_j)

                    max_base_idx = max(oblique_i.max(), oblique_j.max()) + 1
                    k = (max_base_idx // 4) - 1 if max_base_idx > 4 else 1

                    base_feature_names = reconstruct_feature_names(k, context_name, game)
                    oblique_feature_names = generate_oblique_features(base_feature_names, oblique_i, oblique_j)

                    feature_names = base_feature_names + oblique_feature_names
                    base_feature_count = len(base_feature_names)

                    print(f"Oblique tree detected (reconstructed):")
                    print(f"  - k (objects per context): {k}")
                    print(f"  - Base features: {len(base_feature_names)}")
                    print(f"  - Oblique features: {len(oblique_feature_names)}")

                else:
                    # No oblique features
                    n_features = tree.n_features_in_
                    k = (n_features // 4) - 1 if n_features > 4 else 1
                    feature_names = reconstruct_feature_names(k, context_name, game)
                    oblique_i = np.array([])
                    oblique_j = np.array([])
                    base_feature_count = len(feature_names)

                    print(f"Regular tree (no oblique, reconstructed):")
                    print(f"  - k (objects per context): {k}")
                    print(f"  - Features: {n_features}")

            # Get oblique indices if they exist
            is_oblique = hasattr(tree, '_is_oblique') and tree._is_oblique
            if is_oblique:
                oblique_i = np.array(tree._oblique_i) if hasattr(tree, '_oblique_i') else np.array([])
                oblique_j = np.array(tree._oblique_j) if hasattr(tree, '_oblique_j') else np.array([])
            else:
                oblique_i = np.array([])
                oblique_j = np.array([])

            # Debug: Show which features are actually used
            print("\nFeature importance analysis:")
            used_features = get_feature_importance_info(tree, feature_names[:tree.n_features_in_])
            if used_features:
                print("  Features with importance > 0:")
                for idx, name, imp in sorted(used_features, key=lambda x: -x[2])[:10]:
                    print(f"    [{idx:3d}] {name:40s} : {imp:.4f}")

            # Show which feature indices are used in tree nodes
            used_indices = get_tree_feature_usage(tree)
            print(f"\n  Feature indices used in tree nodes: {used_indices}")

            # Check if any oblique features are actually used
            if is_oblique and base_feature_count > 0:
                oblique_used = [i for i in used_indices if i >= base_feature_count]
                if oblique_used:
                    print(f"  ✓ Oblique features ARE being used: indices {oblique_used}")
                else:
                    print(f"  ✗ NO oblique features used (only base features)")

            # Generate the tree code
            tree_body_code = extract_tree_body_as_code(
                tree, feature_names[:tree.n_features_in_], tree.classes_, node=0, depth=1
            )

            # Extract metadata
            folder_dir  = os.path.dirname(tree_path)
            folder_name = os.path.basename(folder_dir)

            # Generate script
            script_text = build_final_script(
                config['env_name'],
                config['vecnorm_path'],
                config['focus_file'],
                config['seed'],
                tree_body_code,
                feature_names[:tree.n_features_in_],
                base_feature_count,
                oblique_i, oblique_j, is_oblique
            )

            # Save script
            target_dir = folder_name
            os.makedirs(target_dir, exist_ok=True)
            script_path = os.path.join(target_dir, f"{folder_name}-{context_name}-rules.py")

            with open(script_path, "w") as fh:
                fh.write(script_text)

            print(f"\n✓ Wrote {script_path}")

if __name__ == "__main__":
    main()