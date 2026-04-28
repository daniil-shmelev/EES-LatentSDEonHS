"""Plot generator: produces ``figures/memory.pdf`` (headline integrator-only
Memory-vs-N plot) and ``figures/memory_full_pipeline.pdf`` (companion plot of
the full Latent SDE pipeline; the path-KL closed form contributes its own
O(N) tape, so end-to-end savings are smaller than the integrator-only case).

The compute-parity numbers are reported as a table, so no parity plot is
generated here. The ``plot_parity`` function below remains for ad-hoc
inspection of ``results/compute_parity.csv``.

Usage::

    python scripts/plot_figures.py
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEMORY_CSV = os.path.join(REPO_ROOT, "results", "memory_sweep_integrator.csv")
DEFAULT_FULL_PIPE_CSV = os.path.join(REPO_ROOT, "results", "memory_sweep.csv")
DEFAULT_PARITY_CSV = os.path.join(REPO_ROOT, "results", "compute_parity.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "figures")

# Style
COLORS = {
    "geometric_euler/autograd": "#d62728",
    "geometric_euler/checkpoint": "#ff7f0e",
    "cfees25/reversible": "#1f77b4",
    "geometric_euler": "#d62728",
    "cfees25": "#1f77b4",
}
LABELS = {
    "geometric_euler/autograd": "Geo-Euler + full autograd",
    "geometric_euler/checkpoint": "Geo-Euler + grad. checkpoint",
    "cfees25/reversible": "CFEES + reversible adjoint",
    "geometric_euler": "Geo-Euler",
    "cfees25": "CFEES",
}


def _to_float(s):
    if s is None or s == "" or s == "n/a":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def plot_memory(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"  no memory sweep CSV at {csv_path}, skipping memory plot")
        return
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    grouped: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
    for r in rows:
        key = f"{r['kind']}/{r['adjoint']}"
        grouped[key].append((int(r["N"]), _to_float(r["peak_alloc_mib"]), r["status"]))

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for key, entries in grouped.items():
        entries.sort(key=lambda x: x[0])
        Ns_ok = [e[0] for e in entries if e[2] == "ok"]
        mibs_ok = [e[1] for e in entries if e[2] == "ok"]
        if not Ns_ok:
            continue
        ax.plot(Ns_ok, mibs_ok, marker="o", color=COLORS.get(key, "k"),
                label=LABELS.get(key, key), linewidth=2)
        for N, mib, status in entries:
            if status == "oom":
                ax.annotate("OOM", xy=(N, max(mibs_ok) if mibs_ok else 1e3),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", color=COLORS.get(key, "k"), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Integrator steps $N$")
    ax.set_ylabel("Peak GPU memory (MiB)")
    ax.set_title(r"SDE integrator memory vs. $N$ -- $S^{15}$, batch=64")
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_parity(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        print(f"  no compute-parity CSV at {csv_path}, skipping parity plot")
        return
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    # Aggregate across seeds.
    by_solver_B: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for r in rows:
        by_solver_B[(r["solver"], int(r["B"]))].append(r)

    solvers = sorted(set(r["solver"] for r in rows))
    Bs = sorted(set(int(r["B"]) for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    # Left: ELBO. Right: accuracy.
    for solver in solvers:
        means_elbo, stds_elbo, means_acc, stds_acc = [], [], [], []
        Bs_s = []
        for B in Bs:
            seeds = by_solver_B.get((solver, B), [])
            if not seeds:
                continue
            elbos = [_to_float(s.get("test_elbo")) for s in seeds]
            elbos = [v for v in elbos if not math.isnan(v)]
            accs = [_to_float(s.get("test_acc")) for s in seeds]
            accs = [v for v in accs if not math.isnan(v)]
            if elbos and accs:
                Bs_s.append(B)
                means_elbo.append(sum(elbos) / len(elbos))
                stds_elbo.append(_std(elbos))
                means_acc.append(sum(accs) / len(accs))
                stds_acc.append(_std(accs))
        c = COLORS.get(solver, "k")
        lbl = LABELS.get(solver, solver)
        axes[0].errorbar(Bs_s, means_elbo, yerr=stds_elbo, marker="o", color=c, label=lbl, linewidth=2)
        axes[1].errorbar(Bs_s, means_acc, yerr=stds_acc, marker="o", color=c, label=lbl, linewidth=2)

    axes[0].set_xlabel("Total NN evals per trajectory $B$")
    axes[0].set_ylabel("Test ELBO")
    axes[0].set_xscale("log")
    axes[0].grid(False)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("Compute-parity ELBO")
    axes[1].set_xlabel("Total NN evals per trajectory $B$")
    axes[1].set_ylabel("Test classification accuracy")
    axes[1].set_xscale("log")
    axes[1].grid(False)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_title("Compute-parity accuracy")

    fig.suptitle("Compute parity at fixed total NN-eval budget -- Human Activity classification",
                 y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-csv", default=DEFAULT_MEMORY_CSV)
    parser.add_argument("--full-pipe-csv", default=DEFAULT_FULL_PIPE_CSV)
    parser.add_argument("--parity-csv", default=DEFAULT_PARITY_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Headline: integrator-only memory plot (mirrors the existing T^7 figure).
    plot_memory(args.memory_csv, os.path.join(args.out_dir, "memory.pdf"))
    # Companion: full-pipeline (with KL closed-form + decoder) -- shows the
    # constant-factor savings achievable in actual Latent SDE training.
    if args.full_pipe_csv != args.memory_csv:
        plot_memory(args.full_pipe_csv, os.path.join(args.out_dir, "memory_full_pipeline.pdf"))
    # Parity is reported as a table in the manuscript -- skip plot generation.


if __name__ == "__main__":
    main()
