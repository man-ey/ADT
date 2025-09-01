import os
import numpy as np
import re
import sys

EVAL_FILENAMES = {"evaluation", "1evaluation.txt", "evaluation.txt"}

def compute_iqm(values):
    """Compute the interquartile mean (IQM) of a list/array of values."""
    values = np.sort(values)
    if values.size == 0:
        return float("nan")
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    trimmed_values = values[(values >= q1) & (values <= q3)]
    return float(np.mean(trimmed_values)) if trimmed_values.size else float("nan")

def extract_values_after_episode_rewards(filepath):
    """Extract float values from lines after the first 'Episode Rewards' line."""
    values = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_index = next((i for i, line in enumerate(lines) if "Episode Rewards" in line), None)
        if start_index is None:
            print(f"[WARNING] 'Episode Rewards' not found in {filepath}")
            return []

        for line in lines[start_index + 1:]:
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)  # also capture integers just in case
            for m in matches:
                try:
                    values.append(float(m))
                except ValueError:
                    pass
    except Exception as e:
        print(f"[ERROR] Failed to extract values from {filepath}: {e}")
    return values

def process_evaluation_file(evaluation_file_path):
    values = extract_values_after_episode_rewards(evaluation_file_path)
    if values:
        iqm = compute_iqm(np.array(values, dtype=float))
        out_dir = os.path.dirname(evaluation_file_path)
        output_path = os.path.join(out_dir, "evaluation_iqm.txt")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"IQM: {iqm}\n")
            print(f"[INFO] Processed {evaluation_file_path} -> IQM: {iqm} -> {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed writing IQM for {evaluation_file_path}: {e}")
    else:
        print(f"[INFO] No valid numeric values found in {evaluation_file_path}")

def main(base_folder):
    if not os.path.isdir(base_folder):
        print(f"[ERROR] Not a directory: {base_folder}")
        return
    found = 0
    for root, _, files in os.walk(base_folder):
        for fname in files:
            if fname in EVAL_FILENAMES:
                eval_path = os.path.join(root, fname)
                process_evaluation_file(eval_path)
                found += 1
    if found == 0:
        print(f"[INFO] No evaluation files found under: {base_folder}")

if __name__ == "__main__":
    # Use CLI arg if provided; otherwise fall back to hardcoded path
    if len(sys.argv) > 1:
        main_folder = sys.argv[1]
    else:
        main_folder = "tree_splitting_by_attention/context_trees"
    main(main_folder)
