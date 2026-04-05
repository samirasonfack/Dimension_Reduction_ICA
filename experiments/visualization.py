"""
Visualization utilities for ICA benchmarks
============================================
Functions to plot:
  - Recovered vs. true source signals
  - Convergence (log-likelihood) curves
  - Amari index bar chart across algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_sources(
    S_true: np.ndarray,
    results: dict,
    n_show: int = 500,
    figsize: tuple = (14, 10),
    save_path: str = None,
):
    """
    Plot true sources alongside each algorithm's recovered components.

    Parameters
    ----------
    S_true : ndarray (n_samples, n_sources)
    results : dict returned by ``run_benchmark``
    n_show : int — number of time steps to display
    save_path : str or None — if given, saves the figure
    """
    algo_names = [k for k in results if not k.startswith("_")]
    n_algo = len(algo_names)
    k = S_true.shape[1]
    t = np.arange(n_show)

    fig, axes = plt.subplots(
        k, n_algo + 1,
        figsize=figsize,
        sharex=True,
    )
    fig.suptitle("True sources vs. recovered components", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    for c in range(k):
        # True source
        ax = axes[c, 0]
        ax.plot(t, S_true[:n_show, c], color=colors[c], linewidth=0.8)
        if c == 0:
            ax.set_title("True sources", fontsize=10)
        ax.set_ylabel(f"s{c+1}", rotation=0, labelpad=15)
        ax.set_yticks([])

        for j, name in enumerate(algo_names):
            ax = axes[c, j + 1]
            S_hat = results[name]["S_hat"]
            # Match component ordering by maximum absolute correlation
            corrs = [abs(np.corrcoef(S_true[:, c], S_hat[:, ci])[0, 1]) for ci in range(k)]
            best = int(np.argmax(corrs))
            ax.plot(t, S_hat[:n_show, best], color=colors[c], linewidth=0.8, alpha=0.85)
            if c == 0:
                ax.set_title(name, fontsize=9)
            ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_convergence(
    results: dict,
    figsize: tuple = (9, 4),
    save_path: str = None,
):
    """
    Plot log-likelihood convergence curves for algorithms that track it.

    Parameters
    ----------
    results : dict returned by ``run_benchmark``
    save_path : str or None
    """
    fig, ax = plt.subplots(figsize=figsize)
    algo_names = [k for k in results if not k.startswith("_")]

    plotted = False
    for name in algo_names:
        curve = results[name].get("loss_curve", [])
        if curve:
            ax.plot(curve, label=name, linewidth=1.5)
            plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "No convergence data available",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Epoch / Iteration")
    ax.set_ylabel("Avg. log-likelihood")
    ax.set_title("Convergence curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_amari_scores(
    results: dict,
    figsize: tuple = (8, 4),
    save_path: str = None,
):
    """
    Bar chart of Amari performance indices (lower is better).

    Parameters
    ----------
    results : dict returned by ``run_benchmark``
    save_path : str or None
    """
    algo_names = [k for k in results if not k.startswith("_")]
    scores = [results[name]["amari"] for name in algo_names]
    times = [results[name]["time_s"] for name in algo_names]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    colors = plt.cm.Set2.colors[: len(algo_names)]

    # Amari index
    x = np.arange(len(algo_names))
    bars = axes[0].bar(x, scores, color=colors, edgecolor="white")
    axes[0].set_title("Amari Index (↓ meilleur)")
    axes[0].set_ylabel("Amari Index")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(algo_names, rotation=25, ha="right", fontsize=9)
    for bar, val in zip(bars, scores):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    # Runtime
    bars2 = axes[1].bar(x, times, color=colors, edgecolor="white")
    axes[1].set_title("Runtime (s)")
    axes[1].set_ylabel("Temps (s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(algo_names, rotation=25, ha="right", fontsize=9)
    for bar, val in zip(bars2, times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}s",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
