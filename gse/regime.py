"""Modèle à changement de régime (Hardy RSLN-2) par EM / Baum-Welch.

Deux usages :
  * marges de référence : une chaîne 2-états PAR actif (separate), pour
    reproduire Parametres_models.xlsx ;
  * cadre de la note : un régime latent COMMUN (joint), émissions gaussiennes
    multivariées, fournissant les probabilités lissées xi_t(a) qui pilotent
    la dépendance par régime et la simulation.

EM robuste : multi-démarrage, planchers de variance, tri des états par
volatilité (anti label-switching), filtre de Hamilton + lisseur de Kim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .preprocessing import Preprocessed


# --------------------------------------------------------------------------- #
def _gauss_logpdf(X, mean, cov):
    """log densité N(mean, cov) ; X (n,D)."""
    D = X.shape[1]
    cov = np.atleast_2d(cov)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(D))
    sol = np.linalg.solve(L, (X - mean).T)
    quad = np.sum(sol ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)


def _forward_backward(logB, P, pi):
    """Filtre de Hamilton (avant, mis à l'échelle) + lisseur de Kim (arrière).

    logB : (n,K) log-densités d'émission. Retourne (gamma, xi_pairs, loglik)
    avec gamma=(n,K) probas lissées, xi_pairs=(K,K) sommes des transitions.
    """
    n, K = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))      # stabilité
    scale_log = logB.max(axis=1)
    a = np.zeros((n, K)); c = np.zeros(n)
    pred = np.zeros((n, K))
    a0 = pi * B[0]; c[0] = a0.sum(); a[0] = a0 / c[0]; pred[0] = pi
    for t in range(1, n):
        pr = a[t - 1] @ P
        pred[t] = pr
        at = pr * B[t]; c[t] = at.sum(); a[t] = at / c[t]
    b = np.zeros((n, K)); b[-1] = 1.0
    for t in range(n - 2, -1, -1):
        b[t] = (P @ (B[t + 1] * b[t + 1])) / c[t + 1]
    gamma = a * b
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(n - 1):
        d = a[t][:, None] * P * (B[t + 1] * b[t + 1])[None, :]
        d /= d.sum()
        xi += d
    loglik = float(np.sum(np.log(c) + scale_log))
    return gamma, xi, loglik, pred


def _em_gaussian_hmm(X, K=2, restarts=12, max_iter=500, tol=1e-8,
                     var_floor=1e-6, seed=0):
    """EM pour HMM gaussien (émissions N(m_a, V_a)). X : (n,D)."""
    rng = np.random.default_rng(seed)
    X = np.atleast_2d(X)
    if X.shape[0] < X.shape[1]:
        X = X.T
    n, D = X.shape
    best = None
    for r in range(restarts):
        # init : quantiles globaux + perturbation
        if r == 0:
            order = np.argsort(X[:, 0])
            idx = np.array_split(order, K)
            means = np.array([X[i].mean(0) for i in idx])
            covs = np.array([np.cov(X[i].T, ddof=0).reshape(D, D)
                             + var_floor * np.eye(D) for i in idx])
        else:
            sel = rng.choice(n, K, replace=False)
            means = X[sel] + rng.normal(0, X.std(0) * 0.3, (K, D))
            covs = np.array([np.cov(X.T, ddof=0).reshape(D, D)
                             + var_floor * np.eye(D)] * K)
        P = np.full((K, K), 1.0 / K)
        pi = np.full(K, 1.0 / K)
        ll_old = -np.inf
        for _ in range(max_iter):
            logB = np.column_stack([_gauss_logpdf(X, means[k], covs[k])
                                    for k in range(K)])
            gamma, xi, ll, _ = _forward_backward(logB, P, pi)
            P = xi / xi.sum(axis=1, keepdims=True)
            pi = gamma[0].copy()
            for k in range(K):
                w = gamma[:, k]; sw = w.sum()
                means[k] = (w[:, None] * X).sum(0) / sw
                dx = X - means[k]
                covs[k] = (w[:, None, None] * np.einsum('ti,tj->tij', dx, dx)).sum(0) / sw
                covs[k] += var_floor * np.eye(D)
            if abs(ll - ll_old) < tol:
                break
            ll_old = ll
        if best is None or ll > best["ll"]:
            best = dict(ll=ll, means=means.copy(), covs=covs.copy(),
                        P=P.copy(), pi=pi.copy(), gamma=gamma.copy())
    # tri des états par volatilité décroissante (état 0 = forte vol = régime 1)
    vol = np.array([np.sqrt(np.trace(best["covs"][k]) / D) for k in range(K)])
    o = np.argsort(vol)[::-1]
    best["means"] = best["means"][o]
    best["covs"] = best["covs"][o]
    best["P"] = best["P"][np.ix_(o, o)]
    best["pi"] = best["pi"][o]
    best["gamma"] = best["gamma"][:, o]
    return best


# --------------------------------------------------------------------------- #
def select_k(returns_df, k_max=4, restarts=12, var_floor=1e-6):
    """Analyse du choix du nombre d'états latents (régime commun, joint).

    Ajuste le HMM gaussien multivarié pour K=1..k_max et renvoie, par K, la
    log-vraisemblance optimale, le nombre de paramètres libres, AIC et BIC,
    ainsi que K* = argmin BIC.
    """
    X = returns_df.dropna().values
    n, D = X.shape
    rows = []
    for K in range(1, k_max + 1):
        fit = _em_gaussian_hmm(X, K=K, restarts=restarts, var_floor=var_floor, seed=100 + K)
        p = K * D + K * D * (D + 1) // 2 + K * (K - 1) + (K - 1)
        ll = fit["ll"]
        rows.append(dict(K=K, loglik=ll, p=p,
                         AIC=-2 * ll + 2 * p, BIC=-2 * ll + p * np.log(n)))
    kstar = min(rows, key=lambda r: r["BIC"])["K"]
    return dict(n=n, D=D, table=rows, k_star=kstar)


@dataclass
class RegimeFitResult:
    name: str
    model: str = "RSLN2"
    params_separate: dict = field(default_factory=dict)   # par actif (réf.)
    joint: dict = field(default_factory=dict)             # régime commun
    equity_names: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)


def _annualize_regime(m_month, s_month):
    """(moyenne, vol) mensuelles 100*log -> (drift %, vol %) annualisés."""
    sig = s_month * np.sqrt(12.0)
    mu = 12.0 * m_month + 0.5 * sig ** 2 / 100.0     # drift (Hardy)
    return mu, sig


def _invariant(P):
    """Loi invariante pi (vecteur propre gauche de P pour la valeur propre 1)."""
    w, V = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(V[:, k])
    pi = pi / pi.sum()
    return np.clip(pi, 0, None) / np.clip(pi, 0, None).sum()


def fit_rsln2(pre: Preprocessed, spec: dict, dt: float) -> RegimeFitResult:
    restarts = int(pre.meta.get("em_restarts", 20))
    vf = float(pre.meta.get("var_floor", 1e-6))
    common = bool(pre.meta.get("common_regime", True))
    do_kselect = bool(pre.meta.get("k_select", True))
    k_max = int(pre.meta.get("k_max", 5))
    names = list(pre.data.keys())
    df = pd.concat([pre.data[nm].rename(nm) for nm in names], axis=1).dropna()
    Xj = df.values
    n, D = Xj.shape

    # ---- choix du nombre d'états K* (régime commun) ----
    ks = None
    if common and do_kselect:
        ks = select_k(df, k_max=k_max, restarts=restarts, var_floor=vf)
        K = ks["k_star"]
    else:
        K = int(pre.meta.get("n_states", 2))

    # ---- régime latent COMMUN (joint, émissions multivariées, K états) ----
    jfit = _em_gaussian_hmm(Xj, K=K, restarts=restarts, var_floor=vf, seed=999)
    P = jfit["P"]
    pi = _invariant(P)
    xi = pd.DataFrame(jfit["gamma"], index=df.index,
                      columns=[f"reg{k+1}" for k in range(K)])
    s_month = np.sqrt(np.diagonal(jfit["covs"], axis1=1, axis2=2))   # (K,D)
    # paramètres annualisés par actif et par régime, sous le régime COMMUN
    reg_by_eq = {}
    for j, nm in enumerate(names):
        regs = []
        for k in range(K):
            mu, sig = _annualize_regime(float(jfit["means"][k, j]), float(s_month[k, j]))
            regs.append(dict(mu=mu, sigma=sig,
                             m_month=float(jfit["means"][k, j]),
                             s_month=float(s_month[k, j])))
        reg_by_eq[nm] = regs
    joint = dict(P=P, pi=pi, xi=xi, means=jfit["means"], covs=jfit["covs"],
                 returns=df, loglik=jfit["ll"], K=K, m_month=jfit["means"],
                 s_month=s_month, regimes_by_equity=reg_by_eq, kselect=ks)

    # ---- chaînes SÉPARÉES (uniquement pour comparaison à la référence) ----
    sep = {}
    if bool(pre.meta.get("compare_separate", True)):
        Ksep = 2
        for j, nm in enumerate(names):
            x = pre.data[nm].values.reshape(-1, 1)
            fit = _em_gaussian_hmm(x, K=Ksep, restarts=restarts, var_floor=vf, seed=j)
            regimes = []
            for k in range(Ksep):
                mu, sig = _annualize_regime(float(fit["means"][k, 0]),
                                            float(np.sqrt(fit["covs"][k, 0, 0])))
                regimes.append(dict(mu=mu, sigma=sig,
                                    m_month=float(fit["means"][k, 0]),
                                    s_month=float(np.sqrt(fit["covs"][k, 0, 0]))))
            Ps = fit["P"]
            sep[nm] = dict(regimes=regimes, P=Ps.tolist(),
                           p_1to2=float(Ps[0, 1]), p_1to1=float(Ps[0, 0]),
                           p_2to1=float(Ps[1, 0]), p_2to2=float(Ps[1, 1]),
                           loglik=fit["ll"])

    return RegimeFitResult(pre.name, "RSLN2", sep, joint, names, list(names))
