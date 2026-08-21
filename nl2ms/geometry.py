"""Latent-space geometry: global structure, local structure, spectra.

Operates on matrices of shape ``(n_samples, hidden_dim)`` -- one layer at a
time, across the sample population. This is distinct from
:func:`nl2ms.signals.trajectory_metrics`, which works within a single sample
across layers.

A recurring caution applies to everything in this module: residual-stream
norm and anisotropy both grow with depth in decoder-only transformers as a
matter of ordinary architecture, independent of any reasoning. Every measure
here is therefore reported alongside a scale-invariant counterpart wherever
one exists, so a depth trend is not mistaken for a transition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .signals import EPS, cosine_similarity_matrix, derivative

# ---------------------------------------------------------------------------
# Spectral summaries
# ---------------------------------------------------------------------------
def singular_spectrum(X: np.ndarray, center: bool = True,
                      max_components: Optional[int] = None) -> Dict[str, np.ndarray]:
    """SVD summary of a layer's representation cloud.

    Centering is on by default: without it the (large, depth-growing) mean
    residual vector dominates the first singular direction and every layer
    looks rank-1.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return {"singular_values": np.array([]), "explained_variance_ratio": np.array([]),
                "n_samples": int(X.shape[0]) if X.ndim == 2 else 0}
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    try:
        s = np.linalg.svd(Xc, compute_uv=False)
    except np.linalg.LinAlgError:
        return {"singular_values": np.array([]), "explained_variance_ratio": np.array([]),
                "n_samples": int(X.shape[0]), "error": "svd_did_not_converge"}
    var = s ** 2
    total = float(np.sum(var))
    ratio = var / total if total > EPS else np.zeros_like(var)
    if max_components:
        s, ratio = s[:max_components], ratio[:max_components]
    return {
        "singular_values": s,
        "explained_variance_ratio": ratio,
        "cumulative_explained_variance": np.cumsum(ratio),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "centered": bool(center),
    }


def effective_rank(singular_values: np.ndarray, eps: float = 1e-12) -> float:
    """Roy & Vetterli effective rank: exp(H(normalised spectrum)).

    A continuous, basis-free stand-in for "how many directions matter",
    equal to n for a flat spectrum and 1 for a rank-1 cloud.
    """
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > eps]
    if s.size == 0:
        return 0.0
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def participation_ratio(singular_values: np.ndarray) -> float:
    """PR = (sum s^2)^2 / sum s^4, on eigenvalues of the covariance."""
    lam = np.asarray(singular_values, dtype=np.float64) ** 2
    lam = lam[lam > EPS]
    if lam.size == 0:
        return 0.0
    return float((np.sum(lam) ** 2) / np.sum(lam ** 2))


def variance_explained_at(ratio: np.ndarray, k: int) -> float:
    r = np.asarray(ratio, dtype=np.float64)
    if r.size == 0:
        return float("nan")
    return float(np.sum(r[:k]))


def components_for_variance(ratio: np.ndarray, target: float = 0.9) -> int:
    """Smallest k whose cumulative explained variance reaches ``target``."""
    r = np.asarray(ratio, dtype=np.float64)
    if r.size == 0:
        return 0
    cum = np.cumsum(r)
    idx = np.searchsorted(cum, target) + 1
    return int(min(idx, r.size))


# ---------------------------------------------------------------------------
# Intrinsic dimension
# ---------------------------------------------------------------------------
def two_nn_intrinsic_dimension(X: np.ndarray, discard_fraction: float = 0.1
                               ) -> Dict[str, float]:
    """TwoNN estimator (Facco et al. 2017).

    Uses only the ratio of the two nearest-neighbour distances, so it is
    insensitive to the overall scale growth of the residual stream -- the
    property that makes it usable across layers where a norm-based estimate
    would drift. Requires >= 10 points to be meaningful.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 10:
        return {"intrinsic_dimension": float("nan"), "n_samples": n,
                "status": "too_few_samples"}
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=3)  # self + 2 neighbours
    r1, r2 = dists[:, 1], dists[:, 2]
    valid = (r1 > EPS) & (r2 > r1)
    if valid.sum() < 10:
        return {"intrinsic_dimension": float("nan"), "n_samples": n,
                "status": "degenerate_neighbour_distances"}
    mu = np.sort(r2[valid] / r1[valid])
    m = mu.size
    keep = int(m * (1 - discard_fraction))
    mu = mu[:keep]
    # F(mu) = 1 - mu^{-d}; regress -log(1 - F) on log(mu) through the origin.
    F = np.arange(1, mu.size + 1, dtype=np.float64) / m
    x = np.log(mu)
    y = -np.log(np.maximum(1.0 - F, EPS))
    denom = float(np.sum(x * x))
    if denom < EPS:
        return {"intrinsic_dimension": float("nan"), "n_samples": n,
                "status": "degenerate_regression"}
    d = float(np.sum(x * y) / denom)
    return {"intrinsic_dimension": d, "n_samples": n, "n_used": int(mu.size),
            "status": "ok"}


# ---------------------------------------------------------------------------
# Global geometry of one layer
# ---------------------------------------------------------------------------
def layer_geometry(X: np.ndarray, labels: Optional[np.ndarray] = None, *,
                   max_components: int = 64,
                   compute_intrinsic_dim: bool = True) -> Dict[str, Any]:
    """Global geometric summary of one layer's representation cloud.

    ``labels`` (optional, boolean or integer) enables within/between-class
    distance statistics, used for the correct-vs-incorrect comparison.
    """
    X = np.asarray(X, dtype=np.float64)
    out: Dict[str, Any] = {"n_samples": int(X.shape[0])}
    if X.ndim != 2 or X.shape[0] < 2:
        out["status"] = "insufficient_samples"
        return out

    norms = np.linalg.norm(X, axis=1)
    out["norm_mean"] = float(np.mean(norms))
    out["norm_std"] = float(np.std(norms))
    out["norm_median"] = float(np.median(norms))

    spec = singular_spectrum(X, center=True, max_components=max_components)
    sv = spec["singular_values"]
    out["singular_values"] = sv
    out["explained_variance_ratio"] = spec["explained_variance_ratio"]
    out["effective_rank"] = effective_rank(sv)
    out["participation_ratio"] = participation_ratio(sv)
    out["pc1_explained_variance"] = variance_explained_at(spec["explained_variance_ratio"], 1)
    out["pc10_explained_variance"] = variance_explained_at(spec["explained_variance_ratio"], 10)
    out["n_components_90pct"] = components_for_variance(spec["explained_variance_ratio"], 0.9)

    cos = cosine_similarity_matrix(X)
    iu = np.triu_indices(X.shape[0], k=1)
    cos_off = cos[iu]
    out["mean_cosine_similarity"] = float(np.mean(cos_off))
    out["std_cosine_similarity"] = float(np.std(cos_off))
    # Anisotropy in the Ethayarajh sense: mean off-diagonal cosine. High
    # values are the norm in deep transformer layers and are NOT by themselves
    # evidence of a transition.
    out["anisotropy"] = out["mean_cosine_similarity"]

    dists = _pairwise_distances(X)
    d_off = dists[iu]
    out["mean_pairwise_distance"] = float(np.mean(d_off))
    out["std_pairwise_distance"] = float(np.std(d_off))
    # Scale-free: distances all inflate with residual norm, this does not.
    out["distance_cv"] = float(np.std(d_off) / max(np.mean(d_off), EPS))

    cov_trace = float(np.trace(np.cov(X, rowvar=False))) if X.shape[1] < 4096 else float("nan")
    out["covariance_trace"] = cov_trace

    if compute_intrinsic_dim:
        out.update({f"twonn_{k}": v for k, v in
                    two_nn_intrinsic_dimension(X).items()})

    if labels is not None:
        out.update(class_separation(X, labels, dists))
    return out


def _pairwise_distances(X: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(X, metric="euclidean"))


def class_separation(X: np.ndarray, labels: np.ndarray,
                     dists: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Within- vs between-class distance structure.

    Reports the ratio as well as the raw means: the raw values scale with
    residual norm, the ratio does not, so only the ratio is comparable across
    layers.
    """
    X = np.asarray(X, dtype=np.float64)
    labels, valid = _clean_labels(labels)
    classes = sorted(np.unique(labels[valid]).tolist()) if valid.any() else []
    out: Dict[str, Any] = {"n_classes": len(classes),
                           "n_labelled": int(valid.sum())}
    if len(classes) < 2:
        out["status"] = "single_class"
        return out
    # Ungraded samples carry no class and are excluded from every
    # within/between statistic rather than being folded into one group.
    X = X[valid]
    labels = labels[valid]
    dists = None
    if dists is None:
        dists = _pairwise_distances(X)

    within: List[float] = []
    between: List[float] = []
    n = X.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within.append(dists[i, j])
            else:
                between.append(dists[i, j])
    out["within_class_distance_mean"] = float(np.mean(within)) if within else float("nan")
    out["between_class_distance_mean"] = float(np.mean(between)) if between else float("nan")
    if within and between:
        out["separation_ratio"] = float(np.mean(between) / max(np.mean(within), EPS))
    else:
        out["separation_ratio"] = float("nan")

    # Per-class covariance summaries: Cov_correct(l) vs Cov_wrong(l).
    for c in classes:
        mask = labels == c
        if mask.sum() < 2:
            continue
        Xi = X[mask]
        spec = singular_spectrum(Xi, center=True)
        key = f"class_{c}"
        out[f"{key}_n"] = int(mask.sum())
        out[f"{key}_effective_rank"] = effective_rank(spec["singular_values"])
        out[f"{key}_norm_mean"] = float(np.mean(np.linalg.norm(Xi, axis=1)))
        out[f"{key}_covariance_trace"] = float(np.trace(np.cov(Xi, rowvar=False))) \
            if Xi.shape[1] < 4096 else float("nan")
    # Fisher-style discriminability along the class-mean difference.
    if len(classes) == 2:
        a, b = classes
        Xa, Xb = X[labels == a], X[labels == b]
        if Xa.shape[0] >= 2 and Xb.shape[0] >= 2:
            diff = Xa.mean(0) - Xb.mean(0)
            nd = np.linalg.norm(diff)
            if nd > EPS:
                u = diff / nd
                pa, pb = Xa @ u, Xb @ u
                pooled = np.sqrt(0.5 * (np.var(pa, ddof=1) + np.var(pb, ddof=1)))
                out["class_mean_distance"] = float(nd)
                out["fisher_ratio"] = float(nd / max(pooled, EPS))
    return out


def _is_nan(x: Any) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def _clean_labels(labels: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce a mixed label array to ``(int codes, valid mask)``.

    Correctness labels arrive as ``True``/``False``/``None`` in an object
    array; ``np.unique`` cannot sort that mix. Ungraded entries (``None`` or
    NaN) are marked invalid so they are excluded rather than silently grouped
    with the ``False`` class.
    """
    arr = np.asarray(labels, dtype=object).ravel()
    codes = np.full(arr.size, -1, dtype=np.int64)
    valid = np.zeros(arr.size, dtype=bool)
    mapping: Dict[Any, int] = {}
    for i, v in enumerate(arr):
        if v is None or _is_nan(v):
            continue
        key = bool(v) if isinstance(v, (bool, np.bool_)) else v
        if key not in mapping:
            mapping[key] = len(mapping)
        codes[i] = mapping[key]
        valid[i] = True
    return codes, valid


# ---------------------------------------------------------------------------
# Local geometry (protocol section 19)
# ---------------------------------------------------------------------------
def local_geometry(X: np.ndarray, labels: Optional[np.ndarray] = None, *,
                   k: int = 10, cov_neighbours: int = 16) -> Dict[str, Any]:
    """Neighbourhood structure: kNN distances, density, anisotropy, purity.

    The question this addresses is whether the representation manifold
    *reorganises* near a candidate critical layer, as opposed to merely
    translating or rescaling -- which the global measures cannot distinguish.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    out: Dict[str, Any] = {"n_samples": n, "k": k}
    if n < k + 2:
        out["status"] = "too_few_samples"
        return out

    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    kq = min(k + 1, n)
    dists, idxs = tree.query(X, k=kq)
    nn_dists = dists[:, 1:]           # drop self
    nn_idx = idxs[:, 1:]

    out["knn_distance_mean"] = float(np.mean(nn_dists))
    out["knn_distance_std"] = float(np.std(nn_dists))
    out["knn_distance_1st_mean"] = float(np.mean(nn_dists[:, 0]))
    # Scale-free density proxy: ratio of the k-th to the 1st NN distance.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = nn_dists[:, -1] / np.maximum(nn_dists[:, 0], EPS)
    out["knn_distance_ratio_mean"] = float(np.mean(ratio[np.isfinite(ratio)]))

    radius = np.maximum(nn_dists[:, -1], EPS)
    density = nn_dists.shape[1] / (radius ** X.shape[1] if X.shape[1] < 20
                                   else radius)  # avoid overflow in high dim
    out["local_density_mean"] = float(np.mean(density))
    out["local_density_log_mean"] = float(np.mean(np.log(np.maximum(density, EPS))))

    # Local covariance anisotropy: eigenvalue spread within each neighbourhood.
    m = min(cov_neighbours, nn_idx.shape[1])
    anis: List[float] = []
    local_er: List[float] = []
    for i in range(n):
        neigh = X[nn_idx[i, :m]]
        if neigh.shape[0] < 2:
            continue
        spec = singular_spectrum(neigh, center=True)
        sv = spec["singular_values"]
        if sv.size < 2:
            continue
        anis.append(float(sv[0] / max(sv[-1], EPS)))
        local_er.append(effective_rank(sv))
    out["local_anisotropy_mean"] = float(np.mean(anis)) if anis else float("nan")
    out["local_anisotropy_median"] = float(np.median(anis)) if anis else float("nan")
    out["local_effective_rank_mean"] = float(np.mean(local_er)) if local_er else float("nan")

    if labels is not None:
        labels, valid = _clean_labels(labels)
        purity: List[float] = []
        for i in range(n):
            if i >= valid.size or not valid[i]:
                continue
            idxs = nn_idx[i]
            neigh_valid = valid[idxs]
            if neigh_valid.sum() == 0:
                continue
            purity.append(float(np.mean(labels[idxs][neigh_valid] == labels[i])))
        out["neighbourhood_purity"] = float(np.mean(purity)) if purity else float("nan")
        out["neighbourhood_purity_n"] = len(purity)
        # Chance level depends on class balance; reported so purity is
        # interpretable rather than impressive-looking by construction.
        lab_valid = labels[valid]
        if lab_valid.size:
            _, counts = np.unique(lab_valid, return_counts=True)
            freqs = counts / counts.sum()
            out["neighbourhood_purity_chance"] = float(np.sum(freqs ** 2))
    return out


def neighbourhood_stability(X_prev: np.ndarray, X_next: np.ndarray,
                            k: int = 10) -> Dict[str, float]:
    """Jaccard overlap of kNN sets between two layers.

    A direct measure of manifold reorganisation: a layer that translates or
    rescales the cloud preserves neighbour identity (overlap near 1), while a
    layer that genuinely reorders the manifold does not.
    """
    from scipy.spatial import cKDTree
    Xp, Xn = np.asarray(X_prev, float), np.asarray(X_next, float)
    n = Xp.shape[0]
    if n < k + 2 or Xn.shape[0] != n:
        return {"knn_jaccard": float("nan"), "knn_overlap": float("nan"), "n": n}
    kq = min(k + 1, n)
    _, ip = cKDTree(Xp).query(Xp, k=kq)
    _, inn = cKDTree(Xn).query(Xn, k=kq)
    overlaps, jaccards = [], []
    for i in range(n):
        a, b = set(ip[i, 1:].tolist()), set(inn[i, 1:].tolist())
        inter = len(a & b)
        overlaps.append(inter / max(len(a), 1))
        jaccards.append(inter / max(len(a | b), 1))
    return {"knn_jaccard": float(np.mean(jaccards)),
            "knn_overlap": float(np.mean(overlaps)), "n": n}


# ---------------------------------------------------------------------------
# Across-layer assembly
# ---------------------------------------------------------------------------
def geometry_across_layers(H: np.ndarray, labels: Optional[np.ndarray] = None, *,
                           cfg: Any = None, layer_indices: Optional[Sequence[int]] = None
                           ) -> Dict[str, Any]:
    """Run the geometry suite at every layer.

    ``H``: ``(n_samples, n_layers, hidden_dim)``.

    Returns per-layer scalar profiles plus the raw per-layer dictionaries.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 3:
        raise ValueError(f"expected (n_samples, n_layers, hidden), got {H.shape}")
    n_samples, n_layers, _ = H.shape

    max_pairwise = getattr(cfg, "max_samples_for_pairwise", 256) if cfg else 256
    knn_k = getattr(cfg, "knn_k", 10) if cfg else 10
    pca_max = getattr(cfg, "pca_max_components", 64) if cfg else 64
    cov_neigh = getattr(cfg, "local_cov_neighbours", 16) if cfg else 16
    do_id = getattr(cfg, "intrinsic_dim_enabled", True) if cfg else True

    if n_samples > max_pairwise:
        rng = np.random.default_rng(0)
        sel = rng.choice(n_samples, size=max_pairwise, replace=False)
        H = H[sel]
        labels = None if labels is None else np.asarray(labels)[sel]
        n_samples = max_pairwise

    layers = list(layer_indices) if layer_indices is not None else list(range(n_layers))
    per_layer: List[Dict[str, Any]] = []
    local_layer: List[Dict[str, Any]] = []
    stability: List[Dict[str, float]] = []

    for li, l in enumerate(layers):
        X = H[:, l, :]
        per_layer.append(layer_geometry(X, labels, max_components=pca_max,
                                        compute_intrinsic_dim=do_id))
        local_layer.append(local_geometry(X, labels, k=knn_k,
                                          cov_neighbours=cov_neigh))
        if li > 0:
            stability.append(neighbourhood_stability(H[:, layers[li - 1], :], X, k=knn_k))
        else:
            stability.append({"knn_jaccard": float("nan"),
                              "knn_overlap": float("nan"), "n": n_samples})

    def profile(records: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
        return np.array([float(r.get(key, np.nan)) if _scalar(r.get(key))
                         else np.nan for r in records], dtype=np.float64)

    profiles: Dict[str, np.ndarray] = {}
    for key in ["norm_mean", "effective_rank", "participation_ratio",
                "pc1_explained_variance", "n_components_90pct",
                "mean_cosine_similarity", "anisotropy", "mean_pairwise_distance",
                "distance_cv", "covariance_trace", "twonn_intrinsic_dimension",
                "within_class_distance_mean", "between_class_distance_mean",
                "separation_ratio", "fisher_ratio", "class_mean_distance"]:
        profiles[key] = profile(per_layer, key)
    for key in ["knn_distance_mean", "knn_distance_ratio_mean",
                "local_density_log_mean", "local_anisotropy_mean",
                "local_effective_rank_mean", "neighbourhood_purity"]:
        profiles[key] = profile(local_layer, key)
    profiles["knn_jaccard"] = np.array([s.get("knn_jaccard", np.nan) for s in stability])
    # Reorganisation rate: 1 - neighbour overlap. Peaks where the manifold
    # actually reorders, rather than where it merely grows.
    profiles["neighbourhood_reorganisation"] = 1.0 - profiles["knn_jaccard"]
    profiles["effective_rank_delta"] = derivative(profiles["effective_rank"])

    return {
        "layers": np.array(layers),
        "n_samples_used": n_samples,
        "profiles": profiles,
        "per_layer": per_layer,
        "local_per_layer": local_layer,
        "stability": stability,
    }


def _scalar(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def pca_trajectory(H: np.ndarray, n_components: int = 3,
                   fit_layers: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Project every layer into one shared PCA basis.

    A shared basis is essential: fitting a separate PCA per layer produces
    axes that are not comparable, and any resulting "trajectory" would be an
    artefact of independently rotating bases rather than a real path.
    """
    H = np.asarray(H, dtype=np.float64)
    n_samples, n_layers, d = H.shape
    flat = H.reshape(-1, d)
    fit_data = flat if fit_layers is None else H[:, list(fit_layers), :].reshape(-1, d)
    mean = fit_data.mean(axis=0, keepdims=True)
    try:
        _, s, vt = np.linalg.svd(fit_data - mean, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"status": "svd_failed"}
    k = min(n_components, vt.shape[0])
    basis = vt[:k]
    proj = (flat - mean) @ basis.T
    var = s ** 2
    total = float(np.sum(var))
    return {
        "projection": proj.reshape(n_samples, n_layers, k),
        "basis": basis,
        "mean": mean,
        "explained_variance_ratio": (var[:k] / total) if total > EPS else np.zeros(k),
        "n_components": k,
        "status": "ok",
    }
