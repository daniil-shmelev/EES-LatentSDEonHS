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
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MEMORY_CSV = os.path.join(REPO_ROOT, "results", "memory_sweep_integrator.csv")
DEFAULT_FULL_PIPE_CSV = os.path.join(REPO_ROOT, "results", "memory_sweep.csv")
DEFAULT_PARITY_CSV = os.path.join(REPO_ROOT, "results", "compute_parity.csv")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "figures")

# Headline-figure style: matches the RNA torus memory plot
# (experiments/rna/plots.py::fig_torus_scaling) so the two manifold-SDE
# memory figures share a visual language.
SCALING_MODE_LABEL = {
    "cfees25/reversible": r"CF-EES(2,5) + Reversible Adj.",
    "geometric_euler/autograd": r"Geo-Euler + Full Adj.",
    "geometric_euler/checkpoint": r"Geo-Euler + Recursive Adj.",
}
SCALING_MODE_COLOR = {
    "cfees25/reversible": "#d62728",
    "geometric_euler/autograd": "#1f77b4",
    "geometric_euler/checkpoint": "#1f77b4",
}
SCALING_MODE_MARKER = {
    "cfees25/reversible": "o",
    "geometric_euler/autograd": "s",
    "geometric_euler/checkpoint": "D",
}
SCALING_MODE_LINESTYLE = {
    "cfees25/reversible": "-",
    "geometric_euler/autograd": "-",
    "geometric_euler/checkpoint": "--",
}


def _set_stix_params(small: int = 8, medium: int = 9, bigger: int = 10) -> None:
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rc("font", size=small)
    plt.rc("axes", titlesize=bigger)
    plt.rc("axes", labelsize=medium)
    plt.rc("xtick", labelsize=small)
    plt.rc("ytick", labelsize=small)
    plt.rc("legend", fontsize=small)
    plt.rc("figure", titlesize=bigger)


def _to_float(s):
    if s is None or s == "" or s == "n/a":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def plot_memory(csv_path: str, out_path: str) -> None:
    """Memory-vs-N plot in the same style as the RNA-torus Figure 1
    (`experiments/rna/plots.py::fig_torus_scaling`).

    Plots absolute peak GPU memory in MiB on log-log axes. Reference
    slopes O(n), O(sqrt(n)) (where applicable) and O(1) annotations
    are anchored to the rightmost point of each curve.
    """
    if not os.path.exists(csv_path):
        print(f"  no memory sweep CSV at {csv_path}, skipping memory plot")
        return
    _set_stix_params(8, 9, 10)
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    grouped: Dict[str, List[Tuple[int, float, str]]] = defaultdict(list)
    for r in rows:
        key = f"{r['kind']}/{r['adjoint']}"
        grouped[key].append((int(r["N"]), _to_float(r["peak_alloc_mib"]), r["status"]))

    # Plot order mirrors the RNA Figure 1: CFEES first (so it's not over-drawn),
    # then the linear-tape baselines.
    plot_order = [
        "cfees25/reversible",
        "geometric_euler/autograd",
        "geometric_euler/checkpoint",
    ]

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    plotted = {}
    for key in plot_order:
        if key not in grouped:
            continue
        entries = sorted(grouped[key], key=lambda x: x[0])
        Ns = np.asarray([e[0] for e in entries if e[2] == "ok"], dtype=float)
        mibs = np.asarray([e[1] for e in entries if e[2] == "ok"], dtype=float)
        if Ns.size == 0:
            continue
        plotted[key] = (Ns, mibs)
        ax.plot(
            Ns, mibs,
            marker=SCALING_MODE_MARKER.get(key, "o"),
            markersize=3, linewidth=1.2,
            color=SCALING_MODE_COLOR.get(key, "k"),
            linestyle=SCALING_MODE_LINESTYLE.get(key, "-"),
            label=SCALING_MODE_LABEL.get(key, key),
        )

    # Reference O(n) line anchored to the rightmost point of the
    # geometric-Euler curve (whose theoretical scaling it represents).
    # CFEES gets an O(1) annotation against its own rightmost point.
    if "geometric_euler/autograd" in plotted:
        Ns, mibs = plotted["geometric_euler/autograd"]
        # Anchor the O(n) reference to the slope between the first two large-N
        # points (where the linear regime dominates) rather than to the
        # constant-overhead small-N regime.
        x_anchor, y_anchor = float(Ns[-1]), float(mibs[-1])
        if Ns.size >= 2 and y_anchor > 0.0:
            x_ref = np.asarray([float(Ns[0]), x_anchor])
            y_ref = y_anchor * (x_ref / x_anchor)  # slope 1 in log-log
            ax.plot(x_ref, y_ref, color="#444444", linewidth=1.2,
                    linestyle=":", alpha=1.0, zorder=1)
            ax.annotate(r"$\mathcal{O}(n)$", xy=(x_anchor, y_anchor),
                        xytext=(12, 0), textcoords="offset points",
                        fontsize=9, color="black",
                        ha="left", va="center", annotation_clip=False)
    if "cfees25/reversible" in plotted:
        Ns, mibs = plotted["cfees25/reversible"]
        ax.annotate(r"$\mathcal{O}(1)$",
                    xy=(float(Ns[-1]), float(mibs[-1])),
                    xytext=(12, 0), textcoords="offset points",
                    fontsize=9, color="black",
                    ha="left", va="center", annotation_clip=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"Peak GPU memory (MiB)")
    ax.legend(loc="upper left", frameon=True, fontsize=7)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
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
