"""Calibrage des marges du Groupe A (et briques de base).

Modèles : V2F (Vasicek 2 facteurs), CIR, BK (log-Vasicek / OU), BS.
Chaque calibrateur renvoie un :class:`FitResult` portant les paramètres
(annualisés), les résidus standardisés (PIT, ~N(0,1)) datés, et la carte de
simulation. Conforme à la note : EMV conditionnel exact, formes fermées.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, ncx2

from .preprocessing import Preprocessed


@dataclass
class FitResult:
    name: str
    model: str
    params: dict
    resid: pd.DataFrame                 # colonnes = composantes, index = dates
    sim: dict = field(default_factory=dict)
    components: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Brique OU scalaire (régression = EMV exact), moyenne fixable
# --------------------------------------------------------------------------- #
def fit_ou_scalar(y: np.ndarray, dt: float, fix_mean=None):
    """OU exact : y_{t+1} = beta y_t + c + xi. Retourne dict de paramètres."""
    y = np.asarray(y, float)
    y0, y1 = y[:-1], y[1:]
    if fix_mean is None:
        beta, c = np.polyfit(y0, y1, 1)
        mean = c / (1.0 - beta)
        resid = y1 - (beta * y0 + c)
    else:
        mean = float(fix_mean)
        beta = float(np.sum((y0 - mean) * (y1 - mean)) / np.sum((y0 - mean) ** 2))
        c = mean * (1.0 - beta)
        resid = (y1 - mean) - beta * (y0 - mean)
    beta = float(np.clip(beta, 1e-8, 1 - 1e-10))   # stabilité (kappa>0)
    k = -np.log(beta) / dt
    v = float(np.var(resid, ddof=0))
    sigma = float(np.sqrt(v * (-2.0 * np.log(beta)) / (dt * (1.0 - beta ** 2))))
    return dict(beta=beta, c=c, kappa=k, mean=mean, v_cond=v, sigma=sigma,
                resid=resid)


# --------------------------------------------------------------------------- #
#  V2F — VAR(1) gaussien, EMV exact (cascade équation par équation)
# --------------------------------------------------------------------------- #
def fit_v2f(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    qs = pre.data["short"]
    ql = pre.data["long"]
    fix = spec.get("fix", {}) or {}
    fix_mu = fix.get("mu", None)

    # (i) facteur long : OU autonome (moyenne fixable)
    lon = fit_ou_scalar(ql.values, dt, fix_mean=fix_mu)
    k2, mu, sig2, Vm = lon["kappa"], lon["mean"], lon["sigma"], lon["v_cond"]

    # (ii) facteur court : MCO sur (1, qs_t, ql_t)  (régression bivariée)
    qs0, qs1 = qs.values[:-1], qs.values[1:]
    ql0 = ql.values[:-1]
    X = np.column_stack([np.ones_like(qs0), qs0, ql0])
    coef, *_ = np.linalg.lstsq(X, qs1, rcond=None)
    c_s, phi11, phi12 = coef
    res_s = qs1 - X @ coef
    phi11 = float(np.clip(phi11, 1e-8, 1 - 1e-10))
    k1 = -np.log(phi11) / dt
    Vr = float(np.var(res_s, ddof=0))
    C_mr = float(np.cov(res_s, lon["resid"], ddof=0)[0, 1])
    rho_c = C_mr / np.sqrt(Vr * Vm) if Vr > 0 and Vm > 0 else 0.0

    # sigma1 net de la variance transmise par le facteur long (note, A bis)
    if abs(k1 - k2) > 1e-8:
        B = ((1 - np.exp(-2 * k2 * dt)) / (2 * k2)
             - 2 * (1 - np.exp(-(k1 + k2) * dt)) / (k1 + k2)
             + (1 - np.exp(-2 * k1 * dt)) / (2 * k1))
        transmit = (k1 / (k1 - k2)) ** 2 * sig2 ** 2 * B
    else:
        transmit = 0.0
    s1_sq = max(Vr - transmit, 1e-12) * 2 * k1 / (1 - np.exp(-2 * k1 * dt))
    sig1 = float(np.sqrt(s1_sq))

    # résidus standardisés (composante par composante)
    idx = qs.index[1:]
    eps_s = res_s / np.sqrt(Vr)
    eps_l = lon["resid"] / np.sqrt(Vm)
    resid = pd.DataFrame({f"{pre.name}.short": eps_s,
                          f"{pre.name}.long": eps_l}, index=idx)

    params = dict(kappa_short=k1, kappa_long=k2, sigma_short=sig1,
                  sigma_long=sig2, mu=mu, rho_c=rho_c,
                  phi11=phi11, phi12=float(phi12), phi22=lon["beta"],
                  c_short=float(c_s), c_long=float(lon["c"]),
                  Vr=Vr, Vm=Vm, C_mr=C_mr)
    sim = dict(kind="V2F", phi11=phi11, phi12=float(phi12), phi22=lon["beta"],
               c_short=float(c_s), c_long=float(lon["c"]),
               sd_short=float(np.sqrt(Vr)), sd_long=float(np.sqrt(Vm)),
               x0_short=float(qs.values[-1]), x0_long=float(ql.values[-1]))
    comps = [f"{pre.name}.short", f"{pre.name}.long"]
    return FitResult(pre.name, "V2F", params, resid, sim, comps)


# --------------------------------------------------------------------------- #
#  CIR — init Euler (forme fermée) + EMV chi^2 décentré
# --------------------------------------------------------------------------- #
def fit_cir(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    s = pre.data["level"].values.astype(float)
    s0, s1 = s[:-1], s[1:]

    # init Euler
    w = np.sqrt(np.abs(s0))
    Xr = np.column_stack([1.0 / w, w])
    cc, *_ = np.linalg.lstsq(Xr, (s1 - s0) / w, rcond=None)
    c1, c2 = cc
    k0 = max(-c2 / dt, 1e-3)
    th0 = max(-c1 / c2, 1e-4)
    resid0 = (s1 - s0) / w - Xr @ cc
    sg0 = max(np.sqrt(np.var(resid0, ddof=0) / dt), 1e-3)

    def negll(logp):
        k, th, sg = np.exp(logp)
        if 1 - np.exp(-k * dt) < 1e-12:
            return 1e12
        cchi = 2 * k / (sg ** 2 * (1 - np.exp(-k * dt)))
        df = 4 * k * th / sg ** 2
        nc = 2 * cchi * s0 * np.exp(-k * dt)
        pdf = 2 * cchi * ncx2.pdf(2 * cchi * s1, df, nc)
        return -np.sum(np.log(np.clip(pdf, 1e-300, None)))

    res = minimize(negll, np.log([k0, th0, sg0]), method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-9, maxiter=8000))
    k, th, sg = np.exp(res.x)

    # résidus PIT via la f.d.r. du chi^2 décentré
    cchi = 2 * k / (sg ** 2 * (1 - np.exp(-k * dt)))
    df = 4 * k * th / sg ** 2
    nc = 2 * cchi * s0 * np.exp(-k * dt)
    u = ncx2.cdf(2 * cchi * s1, df, nc)
    eps = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    resid = pd.DataFrame({pre.name: eps}, index=pre.data["level"].index[1:])

    params = dict(kappa=float(k), sigma=float(sg), theta=float(th))
    sim = dict(kind="CIR", kappa=float(k), theta=float(th), sigma=float(sg),
               x0=float(s[-1]), feller=bool(sg ** 2 <= 2 * k * th))
    return FitResult(pre.name, "CIR", params, resid, sim, [pre.name])


# --------------------------------------------------------------------------- #
#  BK — OU exact sur la série de (100*)log-spread
# --------------------------------------------------------------------------- #
def fit_bk(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    y = pre.data["y"]
    fix = spec.get("fix", {}) or {}
    ou = fit_ou_scalar(y.values, dt, fix_mean=fix.get("mu", None))
    eps = ou["resid"] / np.sqrt(ou["v_cond"])
    resid = pd.DataFrame({pre.name: eps}, index=y.index[1:])
    params = dict(kappa=ou["kappa"], sigma=ou["sigma"], mu=ou["mean"])
    sim = dict(kind="BK", beta=ou["beta"], c=ou["c"],
               sd=float(np.sqrt(ou["v_cond"])), x0=float(y.values[-1]),
               transform=pre.meta.get("transform", "none"))
    return FitResult(pre.name, "BK", params, resid, sim, [pre.name])


# --------------------------------------------------------------------------- #
#  BS — moments : moyenne (source brute) + vol (source déslissée)
# --------------------------------------------------------------------------- #
def fit_bs(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    mean_src = pre.meta.get("mean_source", "raw")
    vol_src = pre.meta.get("vol_source", "unsmoothed")
    income = float(pre.meta.get("income_yield", 0.0))

    r_mean = pre.data["ret_raw"] if mean_src == "raw" else pre.data.get("ret_unsmoothed", pre.data["ret_raw"])
    r_vol = pre.data.get("ret_unsmoothed", pre.data["ret_raw"]) if vol_src == "unsmoothed" else pre.data["ret_raw"]

    m_month = float(r_mean.mean())
    s_month = float(r_vol.std(ddof=0))
    mu_ann = 12.0 * m_month + income
    sig_ann = s_month * np.sqrt(12.0)

    # diagnostics de déslissage (smoothing)
    sigma_raw = float(pre.data["ret_raw"].std(ddof=0) * np.sqrt(12.0))
    has_uns = "ret_unsmoothed" in pre.data
    sigma_uns = float(pre.data["ret_unsmoothed"].std(ddof=0) * np.sqrt(12.0)) if has_uns else np.nan
    b = float(pre.meta.get("ar1_b", np.nan))
    var_infl = (sigma_uns / sigma_raw) ** 2 if has_uns else np.nan

    eps = ((r_vol - r_vol.mean()) / s_month)
    resid = pd.DataFrame({pre.name: eps.values}, index=r_vol.index)

    params = dict(mu=mu_ann, sigma=sig_ann, ar1_b=b, income_yield=income,
                  sigma_raw=sigma_raw, sigma_unsmoothed=sigma_uns,
                  var_inflation=var_infl)
    sim = dict(kind="BS", m_month=m_month + income / 12.0, s_month=s_month,
               x0=None)
    return FitResult(pre.name, "BS", params, resid, sim, [pre.name])


# registry
MARGIN_FITTERS = {"V2F": fit_v2f, "CIR": fit_cir, "BK": fit_bk, "BS": fit_bs}
