"""Small, dependency-light statistics helpers (numpy + scipy)."""
import math
import numpy as np
from scipy import stats


def wilson(k, n, z=1.959964):
    """Wilson 95% interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(a, b):
    """Exact McNemar test on paired binary outcomes a (system 1) and b (system 2).
    Returns dict(b01, b10, p): b01 = system 2 correct only, b10 = system 1 correct only."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else float(stats.binomtest(min(n10, n01), n, 0.5).pvalue)
    return {"only_1": n10, "only_2": n01, "discordant": n, "p": min(1.0, p)}


def paired_diff_ci(a, b, reps=10000, seed=7):
    """Paired difference in proportions (b - a) with a percentile bootstrap 95% CI over pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(reps, n))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boot_mean_ci(x, reps=10000, seed=7):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fleiss_kappa(ratings):
    """ratings: array (items x raters) of category labels. Returns Fleiss' kappa (nan if undefined)."""
    r = np.asarray(ratings)
    n_items, n_raters = r.shape
    cats = sorted(set(r.ravel().tolist()))
    counts = np.array([[int((row == c).sum()) for c in cats] for row in r], float)
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return float("nan") if abs(1 - P_e) < 1e-12 else float((P_bar - P_e) / (1 - P_e))


def bonferroni_alpha(alpha=0.05, m=3):
    return alpha / m


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
