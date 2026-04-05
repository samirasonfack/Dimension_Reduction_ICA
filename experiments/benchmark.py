"""
Benchmark — compare all ICA algorithms on synthetic data
==========================================================
Generates a standard BSS (blind source separation) scenario:
  - k independent source signals (super-Gaussian by default)
  - Mixed via a random matrix A
  - Each algorithm recovers the sources from the mixture X

Metrics
-------
Amari Index
    Scale and permutation-invariant measure of separation quality.
    Values close to 0 indicate perfect separation.
    (Amari et al., 1996)

Runtime
    Wall-clock time for ``fit``.
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_sources(
    n_samples: int = 2000,
    n_sources: int = 3,
    source_type: str = "mixed",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random BSS problem.

    Parameters
    ----------
    n_samples : int
    n_sources : int
    source_type : {'super_gaussian', 'sub_gaussian', 'mixed'}
        Distribution of the latent sources.
    random_state : int

    Returns
    -------
    S : ndarray (n_samples, n_sources)   — true sources
    A : ndarray (n_sources, n_sources)  — mixing matrix
    X : ndarray (n_samples, n_sources)  — observed mixtures
    """
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 8 * np.pi, n_samples)

    sources = []
    for i in range(n_sources):
        kind = source_type
        if kind == "mixed":
            kind = ["super_gaussian", "sub_gaussian", "uniform"][i % 3]

        if kind == "super_gaussian":
            # Laplace-like: sign * exponential
            s = rng.laplace(size=n_samples)
        elif kind == "sub_gaussian":
            # Sine wave (sub-Gaussian)
            s = np.sin((i + 1) * t + rng.uniform(0, np.pi))
        else:
            # Uniform (sub-Gaussian)
            s = rng.uniform(-1, 1, size=n_samples)

        sources.append(s)

    S = np.column_stack(sources)
    S = (S - S.mean(axis=0)) / S.std(axis=0)

    A = rng.standard_normal((n_sources, n_sources))
    # Ensure A is well-conditioned
    while np.linalg.cond(A) > 20:
        A = rng.standard_normal((n_sources, n_sources))

    X = S @ A.T
    return S, A, X


# ---------------------------------------------------------------------------
# Amari Index
# ---------------------------------------------------------------------------

def amari_index(W: np.ndarray, A: np.ndarray) -> float:
    """
    Compute the Amari performance index.

    Parameters
    ----------
    W : estimated unmixing matrix (k × k)
    A : true mixing matrix (k × k)

    Returns
    -------
    float in [0, ∞)  — 0 means perfect separation.
    """
    P = W @ A.T   # should be close to a permutation-scale matrix
    k = P.shape[0]

    def _row_error(P):
        P_abs = np.abs(P)
        row_max = P_abs.max(axis=1, keepdims=True)
        return np.sum(P_abs / row_max, axis=1) - 1

    def _col_error(P):
        P_abs = np.abs(P)
        col_max = P_abs.max(axis=0, keepdims=True)
        return np.sum(P_abs / col_max, axis=0) - 1

    err = (_row_error(P).sum() + _col_error(P).sum()) / (2 * k * (k - 1))
    return float(err)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    n_samples: int = 2000,
    n_components: int = 3,
    source_type: str = "mixed",
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run all four ICA algorithms on synthetic data and collect metrics.

    Returns
    -------
    dict with keys: algorithm name → {'amari': float, 'time_s': float,
                                       'S_hat': ndarray, 'loss_curve': list}
    """
    from ica import InfomaxICA, FastICACustom, SGDICA, AdamICA
    from sklearn.decomposition import FastICA as SklearnFastICA

    S, A, X = make_sources(n_samples, n_components, source_type, random_state)
    results = {}

    algorithms = {
        "Infomax (batch)": InfomaxICA(
            n_components=n_components,
            learning_rate=0.01,
            max_iter=500,
            random_state=random_state,
            verbose=False,
        ),
        "FastICA (custom)": FastICACustom(
            n_components=n_components,
            g="logcosh",
            random_state=random_state,
            verbose=False,
        ),
        "FastICA (sklearn)": SklearnFastICA(
            n_components=n_components,
            random_state=random_state,
        ),
        "SGD-ICA": SGDICA(
            n_components=n_components,
            learning_rate=0.01,
            batch_size=64,
            n_epochs=100,
            lr_schedule="cosine",
            random_state=random_state,
            verbose=False,
        ),
        "Adam-ICA": AdamICA(
            n_components=n_components,
            learning_rate=1e-3,
            batch_size=64,
            n_epochs=100,
            random_state=random_state,
            verbose=False,
        ),
    }

    for name, model in algorithms.items():
        t0 = time.perf_counter()
        S_hat = model.fit_transform(X)
        elapsed = time.perf_counter() - t0

        W = getattr(model, "W_", None) or getattr(model, "components_", None)
        ai = amari_index(W, A) if W is not None else float("nan")
        loss = getattr(model, "loss_curve_", [])

        results[name] = {
            "amari": ai,
            "time_s": elapsed,
            "S_hat": S_hat,
            "loss_curve": loss,
        }

        if verbose:
            print(f"  {name:<22} Amari={ai:.4f}  time={elapsed:.3f}s")

    results["_meta"] = {"S": S, "A": A, "X": X}
    return results
