"""Structure de dépendance : matrice de corrélation par régime.

Conforme à la note (Étape 3, cadre général) : Omega(a) est estimée sur les
résidus standardisés, par régime, pondérée par les probabilités lissées
xi_t(a) du régime COMMUN. Un masque de sensibilité S permet de rendre
certaines corrélations insensibles au régime :
    Omega(a)_ij = S_ij * G_reg(a)_ij + (1 - S_ij) * G_pool_ij
puis projection SDP (Higham). Aucune corrélation n'est fixée a priori.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
def nearest_corr_higham(A, tol=1e-8, max_iter=200):
    """Matrice de corrélation SDP la plus proche (Higham 2002)."""
    A = np.asarray(A, float)
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    Y = A.copy(); dS = np.zeros_like(A)
    for _ in range(max_iter):
        R = Y - dS
        # projection sur le cône SDP
        w, V = np.linalg.eigh(0.5 * (R + R.T))
        Xp = (V * np.clip(w, 0, None)) @ V.T
        dS = Xp - R
        # projection sur diagonale unité
        Ynew = Xp.copy()
        np.fill_diagonal(Ynew, 1.0)
        if np.linalg.norm(Ynew - Y, "fro") / max(np.linalg.norm(Y, "fro"), 1e-12) < tol:
            Y = Ynew; break
        Y = Ynew
    w, V = np.linalg.eigh(0.5 * (Y + Y.T))
    Y = (V * np.clip(w, 1e-10, None)) @ V.T
    d = np.sqrt(np.diag(Y))
    Y = Y / np.outer(d, d)
    return 0.5 * (Y + Y.T)


def _weighted_corr(E, w):
    """Corrélation pondérée des colonnes de E (n,d) par poids w (n,)."""
    w = w / w.sum()
    M = (w[:, None] * E).T @ E          # second moment pondéré (centrage déjà fait)
    d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
    return M / np.outer(d, d)


def _pooled_corr(E):
    M = (E.T @ E) / E.shape[0]
    d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
    return M / np.outer(d, d)


# --------------------------------------------------------------------------- #
def build_mask(components, groupB_set, kind):
    """Masque symétrique S (1 = entrée propre au régime)."""
    d = len(components)
    S = np.zeros((d, d))
    if kind == "full":
        S[:] = 1.0
    elif kind == "none":
        S[:] = 0.0
    elif kind == "groupB":
        isB = np.array([c in groupB_set for c in components])
        S = (isB[:, None] | isB[None, :]).astype(float)
    else:
        raise ValueError(f"masque de corrélation inconnu : {kind}")
    np.fill_diagonal(S, 1.0)
    return S


def compute_dependence(margins, regime, cfg):
    """Calcule {label_regime -> Omega_SDP} et les Cholesky pour la simulation.

    margins : liste de FitResult (Groupe A) avec .resid (DataFrame).
    regime  : RegimeFitResult (Groupe B, régime commun) ou None.
    """
    corr_cfg = cfg.get("correlations", {})
    kind = corr_cfg.get("regime_sensitivity", "groupB")
    htol = float(corr_cfg.get("higham_tol", 1e-8))
    hmax = int(corr_cfg.get("higham_max_iter", 200))

    # --- résidus Groupe A (régime-indépendants) ---
    A_resid = pd.concat([m.resid for m in margins], axis=1)
    A_cols = list(A_resid.columns)

    if regime is None:                       # pas de Groupe B : une seule Omega
        E = A_resid.dropna()
        Om = nearest_corr_higham(_pooled_corr(E.values - E.values.mean(0)), htol, hmax)
        Om = pd.DataFrame(Om, index=A_cols, columns=A_cols)
        return dict(components=A_cols, regimes={"reg1": Om},
                    L={"reg1": np.linalg.cholesky(Om.values)},
                    K=1, groupB=[])

    rets = regime.joint["returns"]            # (n,D) rendements actions
    xi = regime.joint["xi"]                   # (n,K) probas lissées communes
    B_cols = list(rets.columns)
    K = xi.shape[1]
    m_month = regime.joint["m_month"]         # (K,D)
    s_month = regime.joint["s_month"]         # (K,D)

    # index commun à A, B et xi
    idx = A_resid.dropna().index
    idx = idx.intersection(rets.index).intersection(xi.index)
    if len(idx) < 24:
        raise ValueError("Échantillon commun trop court pour la dépendance.")
    EA = (A_resid.loc[idx].values - A_resid.loc[idx].values.mean(0))
    Xb = rets.loc[idx].values
    W = xi.loc[idx].values                    # (m,K)

    components = A_cols + B_cols
    groupB = set(B_cols)
    S = build_mask(components, groupB, kind)

    # résidus Groupe B "poolés" (standardisation inconditionnelle)
    EB_pool = (Xb - Xb.mean(0)) / Xb.std(0, ddof=0)
    E_pool = np.column_stack([EA, EB_pool])
    G_pool = _pooled_corr(E_pool)

    regimes, Ls = {}, {}
    for a in range(K):
        # résidus Groupe B standardisés AU régime a
        EB_a = (Xb - m_month[a]) / s_month[a]
        E_a = np.column_stack([EA, EB_a])
        G_reg = _weighted_corr(E_a, W[:, a])
        Om = S * G_reg + (1.0 - S) * G_pool
        np.fill_diagonal(Om, 1.0)
        Om = nearest_corr_higham(Om, htol, hmax)
        lbl = f"reg{a+1}"
        regimes[lbl] = pd.DataFrame(Om, index=components, columns=components)
        Ls[lbl] = np.linalg.cholesky(Om)
    return dict(components=components, regimes=regimes, L=Ls, K=K,
                groupB=B_cols, mask=kind,
                pooled=pd.DataFrame(G_pool, index=components, columns=components))
