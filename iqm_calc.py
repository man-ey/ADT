import os
import numpy as np
import re
import sys

import os
import sys
import re
import numpy as np

FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def compute_iqm(values, lower=0.20, upper=0.80):
    """
Interquantile mean over [lower, upper] (inclusive).
Example: lower=0.20, upper=0.80 for a 20–80 IQM.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    ql, qu = np.quantile(arr, [lower, upper])
    trimmed = arr[(arr >= ql) & (arr <= qu)]
    return float(np.mean(trimmed)) if trimmed.size else float("nan")

def extract_rewards(filepath):
    """
Find the 'rewards: [ ... ]' line (case-insensitive) and return the list of numbers.
Handles lists on one line or wrapped across lines.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        # Find the bracketed list after 'rewards:'
        m = re.search(r"rewards\s*:\s*\[(.*?)\]", content, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            print(f"[WARNING] rewards list not found in {filepath}")
            return []
        bracket_content = m.group(1)
        nums = [float(x) for x in FLOAT_RE.findall(bracket_content)]
        return nums
    except Exception as e:
        print(f"[ERROR] Failed to parse {filepath}: {e}")
        return []

def write_or_append_iqm(filepath, iqm, lower=0.20, upper=0.80):
    """
If an 'IQM (20-80):' line already exists, update it; otherwise append it at the end.
    """
    tag = f"IQM ({int(lower*100)}-{int(upper*100)}):"
    new_line = f"{tag} {iqm}\n"
    try:
        with open(filepath, "r+", encoding="utf-8") as f:
            content = f.read()
            pattern = re.compile(rf"^\s*{re.escape(tag)}\s*.*$", flags=re.MULTILINE)
            if pattern.search(content):
                content = pattern.sub(new_line.strip(), content)
                f.seek(0)
                f.write(content)
                f.truncate()
                print(f"[INFO] Updated IQM in {filepath} -> {iqm}")
            else:
                if not content.endswith("\n"):
                    f.write("\n")
                f.write(new_line)
                print(f"[INFO] Appended IQM to {filepath} -> {iqm}")
    except Exception as e:
        print(f"[ERROR] Failed to write IQM for {filepath}: {e}")

def is_evaluation_filename(name: str) -> bool:
    """Return True if filename contains 'evaluation' anywhere (case-insensitive)."""
    return "evaluation" in name.lower()

def process_file(path):
    rewards = extract_rewards(path)
    if not rewards:
        print(f"[INFO] No rewards parsed from {path}")
        return
    iqm = compute_iqm(rewards, lower=0.20, upper=0.80)
    write_or_append_iqm(path, iqm, lower=0.20, upper=0.80)

def main(base_folder):
    if not os.path.isdir(base_folder):
        print(f"[ERROR] Not a directory: {base_folder}")
        return
    found = 0
    for root, _, files in os.walk(base_folder, topdown=True, followlinks=False):
        for fname in files:
            if is_evaluation_filename(fname):
                found += 1
                process_file(os.path.join(root, fname))
    if found == 0:
        print(f"[INFO] No *evaluation* files found under: {base_folder}")

if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "/home/arch/Documents/gitRepos/ADT/trees"
    main(base)
