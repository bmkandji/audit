"""Calibrage des marges du Groupe A (et briques de base).

Modèles : V2F (Vasicek 2 facteurs), CIR, BK (log-Vasicek / OU), BS.
Chaque calibrateur renvoie un :class:`FitResult` portant les paramètres
(annualisés), les résidus standardisés (PIT, ~N(0,1)) datés, et la carte de
simulation. Conforme à la note : EMV conditionnel exact, formes fermées.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import minimize
from scipy.stats import norm, ncx2

from .preprocessing import Preprocessed

log = logging.getLogger("gse")


def _check_ar1(beta_raw: float, label: str) -> None:
    """Avertit si le coefficient AR(1) sort de (0,1) avant écrêtage.

    Hors de cet intervalle, le retour à la moyenne kappa = -ln(beta)/dt n'est
    pas défini (série non persistante ou anti-persistante) : l'écrêtage produit
    alors un kappa arbitraire. On le signale au lieu de le masquer.
    """
    if not (0.0 < beta_raw < 1.0):
        log.warning("%s : coefficient AR(1) hors (0,1) (beta=%.4f) ; écrêté pour "
                    "la stabilité — retour à la moyenne peu fiable sur cette série.",
                    label, beta_raw)


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
    _check_ar1(beta, "OU/BK")
    beta = float(np.clip(beta, 1e-8, 1 - 1e-10))   # stabilité (kappa>0)
    k = -np.log(beta) / dt
    v = float(np.var(resid, ddof=0))
    sigma = float(np.sqrt(v * (-2.0 * np.log(beta)) / (dt * (1.0 - beta ** 2))))
    return dict(beta=beta, c=c, kappa=k, mean=mean, v_cond=v, sigma=sigma,
                resid=resid)


# --------------------------------------------------------------------------- #
#  V2F — trois méthodes de calibrage sélectionnables (method:)
#    mle            : EMV joint PUR sur la loi gaussienne exacte (court & long
#                     estimés ENSEMBLE ; Phi, Sigma_cond paramétrés par les
#                     paramètres structurels, sans passage par MCO).
#    ols            : MCO en cascade (réduit-forme : long OU, court régressé).
#    distributional : cibles distributionnelles (Phase 1 ; V2F uniquement).
# --------------------------------------------------------------------------- #
def _v2f_matrices(k1, k2, s1, s2, rho, dt):
    """(Phi=e^{-A dt}, Sigma_cond, Sigma_inf) pour le V2F structurel.

    A = [[k1,-k1],[0,k2]] ; Sigma_w = [[s1^2, rho s1 s2],[rho s1 s2, s2^2]].
    Sigma_inf résout l'équation de Lyapunov A S + S A^T = Sigma_w ;
    Sigma_cond = Sigma_inf - Phi Sigma_inf Phi^T.
    """
    A = np.array([[k1, -k1], [0.0, k2]])
    Sw = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
    Phi = expm(-A * dt)
    Sinf = solve_continuous_lyapunov(A, Sw)
    Scond = Sinf - Phi @ Sinf @ Phi.T
    return Phi, Scond, Sinf


def _v2f_result(pre, k1, k2, s1, s2, rho, mu, dt, method) -> FitResult:
    """Construit le FitResult V2F à partir des paramètres structurels."""
    qs, ql = pre.data["short"], pre.data["long"]
    Phi, Scond, _ = _v2f_matrices(k1, k2, s1, s2, rho, dt)
    Vr, Vm, Cmr = float(Scond[0, 0]), float(Scond[1, 1]), float(Scond[0, 1])
    rho_c = Cmr / np.sqrt(Vr * Vm) if Vr > 0 and Vm > 0 else 0.0
    phi11, phi12, phi22 = float(Phi[0, 0]), float(Phi[0, 1]), float(Phi[1, 1])
    c = (np.eye(2) - Phi) @ np.array([mu, mu])
    c_s, c_l = float(c[0]), float(c[1])
    qs0, qs1 = qs.values[:-1], qs.values[1:]
    ql0, ql1 = ql.values[:-1], ql.values[1:]
    eps_s = (qs1 - (phi11 * qs0 + phi12 * ql0 + c_s)) / np.sqrt(Vr)
    eps_l = (ql1 - (phi22 * ql0 + c_l)) / np.sqrt(Vm)
    resid = pd.DataFrame({f"{pre.name}.short": eps_s,
                          f"{pre.name}.long": eps_l}, index=qs.index[1:])
    params = dict(kappa_short=float(k1), kappa_long=float(k2),
                  sigma_short=float(s1), sigma_long=float(s2), mu=float(mu),
                  rho=float(rho), rho_c=rho_c, phi11=phi11, phi12=phi12,
                  phi22=phi22, c_short=c_s, c_long=c_l, Vr=Vr, Vm=Vm, C_mr=Cmr,
                  method=method)
    sim = dict(kind="V2F", phi11=phi11, phi12=phi12, phi22=phi22,
               c_short=c_s, c_long=c_l, sd_short=float(np.sqrt(Vr)),
               sd_long=float(np.sqrt(Vm)), x0_short=float(qs.values[-1]),
               x0_long=float(ql.values[-1]))
    comps = [f"{pre.name}.short", f"{pre.name}.long"]
    return FitResult(pre.name, "V2F", params, resid, sim, comps)


def _fit_v2f_ols(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    """MCO en cascade (forme réduite) : long OU autonome puis court régressé."""
    qs, ql = pre.data["short"], pre.data["long"]
    fix_mu = (spec.get("fix", {}) or {}).get("mu", None)
    lon = fit_ou_scalar(ql.values, dt, fix_mean=fix_mu)
    k2, mu, sig2, Vm = lon["kappa"], lon["mean"], lon["sigma"], lon["v_cond"]
    qs0, qs1 = qs.values[:-1], qs.values[1:]
    ql0 = ql.values[:-1]
    X = np.column_stack([np.ones_like(qs0), qs0, ql0])
    coef, *_ = np.linalg.lstsq(X, qs1, rcond=None)
    c_s, phi11, phi12 = coef
    res_s = qs1 - X @ coef
    _check_ar1(phi11, f"{pre.name} (V2F court)")
    phi11 = float(np.clip(phi11, 1e-8, 1 - 1e-10))
    k1 = -np.log(phi11) / dt
    Vr = float(np.var(res_s, ddof=0))
    C_mr = float(np.cov(res_s, lon["resid"], ddof=0)[0, 1])
    rho_c = C_mr / np.sqrt(Vr * Vm) if Vr > 0 and Vm > 0 else 0.0
    # sigma1 net de la variance transmise par le facteur long (approx rho=0)
    if abs(k1 - k2) > 1e-8:
        B = ((1 - np.exp(-2 * k2 * dt)) / (2 * k2)
             - 2 * (1 - np.exp(-(k1 + k2) * dt)) / (k1 + k2)
             + (1 - np.exp(-2 * k1 * dt)) / (2 * k1))
        transmit = (k1 / (k1 - k2)) ** 2 * sig2 ** 2 * B
    else:
        transmit = 0.0
    s1_sq = max(Vr - transmit, 1e-12) * 2 * k1 / (1 - np.exp(-2 * k1 * dt))
    sig1 = float(np.sqrt(s1_sq))
    idx = qs.index[1:]
    resid = pd.DataFrame({f"{pre.name}.short": res_s / np.sqrt(Vr),
                          f"{pre.name}.long": lon["resid"] / np.sqrt(Vm)}, index=idx)
    params = dict(kappa_short=k1, kappa_long=k2, sigma_short=sig1,
                  sigma_long=sig2, mu=mu, rho_c=rho_c,
                  phi11=phi11, phi12=float(phi12), phi22=lon["beta"],
                  c_short=float(c_s), c_long=float(lon["c"]),
                  Vr=Vr, Vm=Vm, C_mr=C_mr, method="ols")
    sim = dict(kind="V2F", phi11=phi11, phi12=float(phi12), phi22=lon["beta"],
               c_short=float(c_s), c_long=float(lon["c"]),
               sd_short=float(np.sqrt(Vr)), sd_long=float(np.sqrt(Vm)),
               x0_short=float(qs.values[-1]), x0_long=float(ql.values[-1]))
    comps = [f"{pre.name}.short", f"{pre.name}.long"]
    return FitResult(pre.name, "V2F", params, resid, sim, comps)


def _fit_v2f_mle(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    """EMV SÉQUENTIEL en deux temps (le facteur long est autonome) :
      (1) facteur LONG seul -> EMV de l'OU scalaire => (kappa_2, sigma_2, mu),
          que l'on FIXE ;
      (2) facteur COURT -> EMV du vecteur (court, long), paramètres du long
          fixés, sur (kappa_1, sigma_1, rho) [vitesse, vol court, corrélation
          court-long]. Optimisation bornée (L-BFGS-B) + multi-démarrage.
    Tout est mené sur le domaine admissible (kappa>0, sigma>0, |rho|<1)."""
    qs = pre.data["short"].values
    ql = pre.data["long"].values
    X0 = np.column_stack([qs[:-1], ql[:-1]])
    X1 = np.column_stack([qs[1:], ql[1:]])
    fix_mu = (spec.get("fix", {}) or {}).get("mu", None)

    # --- Étape 1 : facteur long autonome (OU scalaire = EMV conditionnel) ---
    lon = fit_ou_scalar(ql, dt, fix_mean=fix_mu)
    k2, mu, s2 = lon["kappa"], lon["mean"], lon["sigma"]

    # --- Étape 2 : court + corrélation, (k2, s2, mu) FIXÉS, EMV profilé ---
    def negll(p):
        k1, s1, rho = p
        if abs(k1 - k2) < 1e-6:
            return 1e12
        try:
            Phi, Sc, _ = _v2f_matrices(k1, k2, s1, s2, rho, dt)
        except Exception:
            return 1e12
        det = Sc[0, 0] * Sc[1, 1] - Sc[0, 1] ** 2
        if Sc[0, 0] <= 0 or Sc[1, 1] <= 0 or det <= 0:
            return 1e12
        c = (np.eye(2) - Phi) @ np.array([mu, mu])
        r = X1 - (X0 @ Phi.T + c)
        inv = np.array([[Sc[1, 1], -Sc[0, 1]], [-Sc[0, 1], Sc[0, 0]]]) / det
        quad = np.einsum('ti,ij,tj->t', r, inv, r)
        return 0.5 * (len(r) * np.log((2 * np.pi) ** 2 * det) + quad.sum())

    o = _fit_v2f_ols(pre, spec, dt).params
    bounds = [(0.02, 10.0), (1e-3, 50.0), (-0.98, 0.98)]
    k1_0 = float(np.clip(o["kappa_short"], 0.05, 8))
    s1_0 = float(np.clip(o["sigma_short"], 1e-2, 40))
    starts = [[k1_0, s1_0, float(np.clip(o["rho_c"], -0.9, 0.9))],
              [max(2 * k2, 0.1), s1_0, 0.0],
              [0.5, s1_0, 0.5]]
    best = None
    for x0 in starts:
        try:
            r = minimize(negll, x0, method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    k1, s1, rho = best.x
    return _v2f_result(pre, float(k1), k2, float(s1), s2, float(rho), mu, dt, "mle")


def _fit_v2f_distributional(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    """Cibles distributionnelles (Phase 1, V2F) : cale (k1,k2,s1,s2) pour
    répliquer la vol des variations de taux, la corrélation inter-maturités et
    l'écart-type (corrigé) des niveaux, via la matrice de chargement affine
    B(k1,k2) aux maturités tau. mu fixé (cible) ou estimé par régression."""
    pp = spec.get("preprocessing", {}) or {}
    tau = tuple(float(t) for t in pp.get("maturities", [2.0, 10.0]))
    lv, lc, ls = (spec.get("distrib_weights", [1.0, 1.0, 1.0]) + [1, 1, 1])[:3]
    qs, ql = pre.data["short"], pre.data["long"]
    Y = np.column_stack([qs.values, ql.values])
    dY = np.diff(Y, axis=0)
    m = 2
    sig_dY_h = dY.std(axis=0, ddof=0)
    rho_dY_h = float(np.corrcoef(dY[:, 0], dY[:, 1])[0, 1])
    samp_var = ((Y - Y.mean(0)) ** 2).mean(0)        # variance d'échantillon
    n = len(Y)
    L = np.arange(1, n)
    fix_mu = (spec.get("fix", {}) or {}).get("mu", None)
    mu = float(fix_mu) if fix_mu is not None else float(fit_ou_scalar(ql.values, dt)["mean"])

    def loading(k1, k2):
        B = np.zeros((2, 2))
        for i, t in enumerate(tau):
            B1 = (1 - np.exp(-k1 * t)) / k1
            B2 = k1 / (k1 - k2) * ((1 - np.exp(-k2 * t)) / k2 - (1 - np.exp(-k1 * t)) / k1)
            B[i, 0], B[i, 1] = B1 / t, B2 / t
        return B

    def corrected_level_std(B, Phi, Sinf, SY):
        """Écart-type des niveaux corrigé du biais d'autocorrélation, avec
        l'autocorrélation *du modèle* (bornée, cf. Phase 1)."""
        a, d, b = Phi[0, 0], Phi[1, 1], Phi[0, 1]
        aL, dL = a ** L, d ** L
        offL = b * L * aL / a if abs(a - d) < 1e-9 else b * (aL - dL) / (a - d)
        # Gamma_X(l) = Phi^l Sinf  (entrées en fonction de l)
        G00 = aL * Sinf[0, 0] + offL * Sinf[1, 0]
        G01 = aL * Sinf[0, 1] + offL * Sinf[1, 1]
        G10 = dL * Sinf[1, 0]
        G11 = dL * Sinf[1, 1]
        out = np.empty(2)
        for i in range(2):
            b0, b1 = B[i, 0], B[i, 1]
            gY = b0 * b0 * G00 + b0 * b1 * (G01 + G10) + b1 * b1 * G11   # Gamma_Y(l)_ii
            rho = gY / SY[i, i]
            S = n + 2.0 * np.sum((n - L) * rho)
            factor = max(1.0 - S / n ** 2, 1e-3)
            out[i] = np.sqrt(samp_var[i] / factor)
        return out

    def obj(p):
        k1, k2, s1, s2 = np.exp(p)
        if abs(k1 - k2) < 1e-6:
            return 1e12
        B = loading(k1, k2)
        Phi, _, Sinf = _v2f_matrices(k1, k2, s1, s2, 0.0, dt)   # browniens indép.
        SdX = 2 * Sinf - Phi @ Sinf - Sinf @ Phi.T
        SdY, SY = B @ SdX @ B.T, B @ Sinf @ B.T
        sig_dY_m = np.sqrt(np.clip(np.diag(SdY), 1e-18, None))
        rho_dY_m = SdY[0, 1] / np.sqrt(SdY[0, 0] * SdY[1, 1])
        sig_Y_m = np.sqrt(np.clip(np.diag(SY), 1e-18, None))
        sig_Y_h = corrected_level_std(B, Phi, Sinf, SY)
        e_v = lv / m * np.sum((sig_dY_m / sig_dY_h - 1) ** 2)
        e_c = 2 * lc / (m * (m - 1)) * (rho_dY_m - rho_dY_h) ** 2
        e_s = ls / m * np.sum((sig_Y_m / sig_Y_h - 1) ** 2)
        return e_v + e_c + e_s

    # L'objectif distributionnel est mal conditionné (crête sigma^2/kappa : le
    # facteur court se découple quand kappa_1 grandit, B[.,0]~1/kappa_1 -> 0).
    # On ajoute une RÉGULARISATION ridge faible vers l'échelle MCO des kappa
    # pour lever la dégénérescence (sans quoi kappa glisse vers la borne).
    o = _fit_v2f_ols(pre, spec, dt).params
    lk1, lk2 = np.log(max(o["kappa_short"], 1e-3)), np.log(max(o["kappa_long"], 1e-3))
    reg = float(spec.get("distrib_ridge", 0.05))

    def obj_b(p):
        k2, dk, s1, s2 = p
        k1 = k2 + dk
        val = obj(np.log([k1, k2, s1, s2]))
        val += reg * ((np.log(k1) - lk1) ** 2 + (np.log(k2) - lk2) ** 2)
        return val

    bounds = [(0.02, 5.0), (1e-2, 5.0), (1e-3, 50.0), (1e-3, 50.0)]
    s1_0, s2_0 = float(np.clip(o["sigma_short"], 1e-2, 40)), float(np.clip(o["sigma_long"], 1e-2, 40))
    starts = [[float(np.clip(o["kappa_long"], 0.05, 8)),
               float(np.clip(o["kappa_short"] - o["kappa_long"], 0.05, 8)), s1_0, s2_0],
              [0.2, 0.6, s1_0, s2_0], [0.1, 1.0, 1.5, 1.0]]
    best = None
    for x0 in starts:
        try:
            r = minimize(obj_b, x0, method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        if best is None or r.fun < best.fun:
            best = r
    k2, dk, s1, s2 = best.x
    return _v2f_result(pre, k2 + dk, k2, s1, s2, 0.0, mu, dt, "distributional")


_V2F_METHODS = {"mle": _fit_v2f_mle, "ml": _fit_v2f_mle,
                "ols": _fit_v2f_ols, "mco": _fit_v2f_ols,
                "distributional": _fit_v2f_distributional,
                "distrib": _fit_v2f_distributional, "cibles": _fit_v2f_distributional}


def fit_v2f(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    method = str(spec.get("method", "mle")).lower()
    if method not in _V2F_METHODS:
        raise ValueError(f"{pre.name}: méthode V2F inconnue '{method}' "
                         f"(mle | ols | distributional).")
    return _V2F_METHODS[method](pre, spec, dt)


# --------------------------------------------------------------------------- #
#  CIR — méthode sélectionnable : "mle" (chi^2 décentré, init Euler) ou
#  "ols" (régression d'Euler en forme fermée, sans raffinement EMV).
# --------------------------------------------------------------------------- #
def fit_cir(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    method = str(spec.get("method", "mle")).lower()
    s = pre.data["level"].values.astype(float)
    s0, s1 = s[:-1], s[1:]

    # init Euler (forme fermée)
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

    if method in ("ols", "mco", "euler"):
        k, th, sg = k0, th0, sg0            # régression d'Euler, sans EMV
    else:
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

    params = dict(kappa=float(k), sigma=float(sg), theta=float(th),
                  method=("ols" if method in ("ols", "mco", "euler") else "mle"))
    sim = dict(kind="CIR", kappa=float(k), theta=float(th), sigma=float(sg),
               x0=float(s[-1]), feller=bool(sg ** 2 <= 2 * k * th))
    return FitResult(pre.name, "CIR", params, resid, sim, [pre.name])


# --------------------------------------------------------------------------- #
#  BK — OU/log-Vasicek exact. Pour une marge gaussienne à discrétisation
#  exacte, l'EMV conditionnel COÏNCIDE avec le MCO : "mle" et "ols" donnent
#  le même estimateur en forme fermée (accepté pour cohérence de config).
# --------------------------------------------------------------------------- #
def fit_bk(pre: Preprocessed, spec: dict, dt: float) -> FitResult:
    y = pre.data["y"]
    fix = spec.get("fix", {}) or {}
    method = str(spec.get("method", "mle")).lower()
    ou = fit_ou_scalar(y.values, dt, fix_mean=fix.get("mu", None))
    eps = ou["resid"] / np.sqrt(ou["v_cond"])
    resid = pd.DataFrame({pre.name: eps}, index=y.index[1:])
    params = dict(kappa=ou["kappa"], sigma=ou["sigma"], mu=ou["mean"],
                  method=("ols" if method in ("ols", "mco") else "mle"))
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
