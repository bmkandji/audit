"""Simulation Monte-Carlo de la représentation AR agrégée à régime commun.

À chaque pas : (1) tirage du régime commun via P (avant le rendement, comme à
l'estimation) ; (2) chocs corrélés eps = L(regime) z ; (3) propagation de chaque
facteur par sa carte (V2F gaussien exact, BK log-OU exact, BS/Hardy log-rendement,
CIR par inverse-PIT exact ou Alfonsi E(0)). Une seule loi (copule gaussienne par
régime, Student en option) sous-tend estimation et simulation.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy.stats import ncx2, norm, t as student_t

log = logging.getLogger("gse")


def _draw_next_regime(cur, cumP, u):
    thr = cumP[cur]                      # (n,K)
    return np.clip((u[:, None] >= thr).sum(axis=1), 0, cumP.shape[1] - 1)


def _bk_spread(y, transform):
    """Reconstruit le spread depuis le log-spread y selon l'échelle calibrée :
    transform 'log' -> exp(y) ; 'log100'/'none' (série en 100*ln) -> exp(y/100).
    """
    return np.exp(y) if transform == "log" else np.exp(y / 100.0)


def _make_shocks(rng, reg, Ls, K, d, n_paths, copula, nu):
    """Chocs standardisés eps de corrélation Omega(regime).

    Copule gaussienne (défaut) : eps = L(a) z, marginalement N(0,1).
    Copule de Student : on injecte une dépendance de queue (scatter Omega(a),
    nu ddl) puis on restitue des marges N(0,1) par eps = Phi^{-1}(T_nu(t)),
    préservant les volatilités calibrées de chaque facteur.
    """
    z = rng.standard_normal((n_paths, d))
    eps = np.empty_like(z)
    for k in range(K):
        mk = reg == k
        if mk.any():
            eps[mk] = z[mk] @ Ls[k].T
    if copula == "student":
        g = rng.chisquare(nu, size=(n_paths, 1))
        tvec = eps * np.sqrt(nu / g)                 # marges multivariées t_nu
        u = student_t.cdf(tvec, nu)
        eps = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))  # marges restituées N(0,1)
    return eps


def simulate(cal, n_paths=None, horizon_years=None, seed=None, scheme=None):
    cfg = cal.config
    sc = cfg.get("simulation", {})
    n_paths = int(n_paths or sc.get("n_paths", 5000))
    horizon = int(horizon_years or sc.get("horizon_years", 30))
    seed = int(seed if seed is not None else sc.get("seed", 0))
    scheme = scheme or sc.get("cir_scheme", "inverse_pit")
    copula = sc.get("copula", "gaussian")
    nu = float(sc.get("copula_df", 8))
    dt = float(cfg["data"]["dt"])
    n_steps = horizon * 12
    rng = np.random.default_rng(seed)

    # CIR : le schéma d'Alfonsi n'est positif que sous sigma^2 <= 4*kappa*theta
    # et 1 - kappa*dt/2 > 0. Si la condition est violée (possible après un
    # changement de données), on bascule sur l'inverse-PIT, exact en tout point.
    if scheme == "alfonsi":
        for nm, fr in cal.margins.items():
            s = fr.sim
            if s["kind"] == "CIR" and not (
                    s["sigma"] ** 2 <= 4 * s["kappa"] * s["theta"]
                    and 1 - s["kappa"] * dt / 2 > 0):
                log.warning("CIR %s : condition d'Alfonsi violée ; bascule sur "
                            "inverse_pit (exact).", nm)
                scheme = "inverse_pit"
                break

    dep = cal.dependence
    comps = dep["components"]
    ipos = {c: i for i, c in enumerate(comps)}
    d = len(comps)
    K = dep["K"]
    Ls = [dep["L"][f"reg{k+1}"] for k in range(K)]

    # régime commun
    if cal.regime is not None:
        P = np.asarray(cal.regime.joint["P"], float)
        pi = np.asarray(cal.regime.joint["pi"], float)
    else:
        P = np.array([[1.0]]); pi = np.array([1.0])
    cumP = np.cumsum(P, axis=1)
    reg = rng.choice(K, size=n_paths, p=pi)

    # états initiaux
    state = {}
    out = {}
    for nm, fr in cal.margins.items():
        s = fr.sim
        if s["kind"] == "V2F":
            state[nm] = dict(qs=np.full(n_paths, s["x0_short"]),
                             ql=np.full(n_paths, s["x0_long"]))
            out[nm + ".short"] = np.empty((n_paths, n_steps + 1))
            out[nm + ".long"] = np.empty((n_paths, n_steps + 1))
            out[nm + ".short"][:, 0] = s["x0_short"]
            out[nm + ".long"][:, 0] = s["x0_long"]
        elif s["kind"] == "CIR":
            state[nm] = np.full(n_paths, s["x0"])
            out[nm] = np.empty((n_paths, n_steps + 1)); out[nm][:, 0] = s["x0"]
        elif s["kind"] == "BK":
            state[nm] = np.full(n_paths, s["x0"])
            out[nm + ".y"] = np.empty((n_paths, n_steps + 1)); out[nm + ".y"][:, 0] = s["x0"]
            out[nm + ".spread"] = np.empty((n_paths, n_steps + 1))
            out[nm + ".spread"][:, 0] = _bk_spread(s["x0"], s.get("transform", "none"))
        elif s["kind"] == "BS":
            out[nm] = np.empty((n_paths, n_steps + 1)); out[nm][:, 0] = 1.0

    eq = cal.regime
    if eq is not None:
        eq_names = eq.equity_names
        m_reg = eq.joint["m_month"]          # (K,D)
        s_reg = eq.joint["s_month"]          # (K,D)
        for nm in eq_names:
            out[nm] = np.empty((n_paths, n_steps + 1)); out[nm][:, 0] = 1.0

    reg_path = np.empty((n_paths, n_steps + 1), dtype=int); reg_path[:, 0] = reg

    for t in range(1, n_steps + 1):
        # Régime gouvernant le rendement du pas t : E_t, tiré AVANT le rendement
        # (même convention qu'à l'estimation, où l'émission x_t est sous E_t).
        u = rng.random(n_paths)
        reg = _draw_next_regime(reg, cumP, u)
        reg_path[:, t] = reg

        # chocs corrélés par régime (copule gaussienne par défaut, Student en option)
        eps = _make_shocks(rng, reg, Ls, K, d, n_paths, copula, nu)

        # --- Groupe A ---
        for nm, fr in cal.margins.items():
            s = fr.sim
            if s["kind"] == "V2F":
                qs, ql = state[nm]["qs"], state[nm]["ql"]
                e_s = eps[:, ipos[nm + ".short"]]; e_l = eps[:, ipos[nm + ".long"]]
                ql_new = s["c_long"] + s["phi22"] * ql + s["sd_long"] * e_l
                qs_new = s["c_short"] + s["phi11"] * qs + s["phi12"] * ql + s["sd_short"] * e_s
                state[nm]["qs"], state[nm]["ql"] = qs_new, ql_new
                out[nm + ".short"][:, t] = qs_new
                out[nm + ".long"][:, t] = ql_new
            elif s["kind"] == "CIR":
                x = np.maximum(state[nm], 0.0)
                e = eps[:, ipos[nm]]
                k, th, sg = s["kappa"], s["theta"], s["sigma"]
                if scheme == "inverse_pit":
                    cchi = 2 * k / (sg ** 2 * (1 - np.exp(-k * dt)))
                    df = 4 * k * th / sg ** 2
                    nc = 2 * cchi * x * np.exp(-k * dt)
                    xn = ncx2.ppf(np.clip(norm.cdf(e), 1e-6, 1 - 1e-6), df, nc) / (2 * cchi)
                else:  # Alfonsi E(0)
                    a = 1 - k * dt / 2.0
                    dW = np.sqrt(dt) * e
                    xn = (a * np.sqrt(x) + sg * dW / (2 * a)) ** 2 + (k * th - sg ** 2 / 4.0) * dt
                xn = np.maximum(xn, 0.0)
                state[nm] = xn; out[nm][:, t] = xn
            elif s["kind"] == "BK":
                y = state[nm]; e = eps[:, ipos[nm]]
                yn = s["c"] + s["beta"] * y + s["sd"] * e
                state[nm] = yn
                out[nm + ".y"][:, t] = yn
                out[nm + ".spread"][:, t] = _bk_spread(yn, s.get("transform", "none"))
            elif s["kind"] == "BS":
                e = eps[:, ipos[nm]]
                r = s["m_month"] + s["s_month"] * e
                out[nm][:, t] = out[nm][:, t - 1] * np.exp(r / 100.0)

        # --- Groupe B (actions, régime commun) ---
        if eq is not None:
            for j, nm in enumerate(eq_names):
                e = eps[:, ipos[nm]]
                r = m_reg[reg, j] + s_reg[reg, j] * e
                out[nm][:, t] = out[nm][:, t - 1] * np.exp(r / 100.0)

    out["_regime"] = reg_path
    return out


def summarize(sim_out, dt=1 / 12):
    """Statistiques de contrôle : vol annualisée simulée par sortie."""
    rows = []
    for k, arr in sim_out.items():
        if k.startswith("_"):
            continue
        x = arr
        if (x > 0).all() and not k.endswith(".short") and not k.endswith(".long") and ".y" not in k:
            r = np.diff(np.log(x), axis=1)
            vol = float(np.nanstd(r) / np.sqrt(dt) * 100)
            term = float(np.nanmean(x[:, -1]))
            rows.append((k, "log-vol%/an", round(vol, 2), "E[terminal]", round(term, 4)))
        else:
            vol = float(np.nanstd(np.diff(x, axis=1)) / np.sqrt(dt))
            rows.append((k, "vol(diff)/an", round(vol, 4), "E[terminal]", round(float(np.nanmean(x[:, -1])), 4)))
    return pd.DataFrame(rows, columns=["sortie", "m1", "v1", "m2", "v2"])
