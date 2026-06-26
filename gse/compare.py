"""Comparaison des paramètres calibrés avec la référence Parametres_models.xlsx.

Inclut aussi les diagnostics d'adéquation des résidus PIT (normalité/queue),
support de l'hypothèse de copule gaussienne : ils signalent automatiquement,
sur tout nouveau jeu de données, un facteur dont la marge standardisée s'écarte
de la normalité.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kstest, kurtosis

# (ligne 0-based, colonne 2026=3) -> valeur de référence
_REF_CELLS = {
    ("inflation", "kappa_long"): (5, 3),
    ("inflation", "sigma_long"): (6, 3),
    ("inflation", "mu"): (7, 3),
    ("inflation", "kappa_short"): (9, 3),
    ("inflation", "sigma_short"): (10, 3),
    ("real_rate", "kappa_long"): (12, 3),
    ("real_rate", "sigma_long"): (13, 3),
    ("real_rate", "mu"): (14, 3),
    ("real_rate", "kappa_short"): (30, 3),
    ("real_rate", "sigma_short"): (31, 3),
    ("credit", "kappa"): (26, 3),
    ("credit", "sigma"): (27, 3),
    ("credit", "theta"): (28, 3),
    ("dette_privee", "kappa"): (46, 3),
    ("dette_privee", "sigma"): (47, 3),
    ("dette_privee", "mu"): (48, 3),
    ("pe", "mu"): (38, 3),
    ("pe", "sigma"): (39, 3),
    ("infra", "mu"): (21, 3),
    ("infra", "sigma"): (22, 3),
    ("immobilier", "mu"): (23, 3),
    ("immobilier", "sigma"): (24, 3),
    # Actions : seules les caractéristiques de régime (moyenne/vol) sont
    # comparées. La chaîne étant COMMUNE, il n'y a pas de matrice de
    # transition par actif à comparer (réf. = chaînes séparées).
    ("equities:Action_EUR", "R1.mu"): (16, 3),
    ("equities:Action_EUR", "R1.sigma"): (17, 3),
    ("equities:Action_EUR", "R2.mu"): (18, 3),
    ("equities:Action_EUR", "R2.sigma"): (19, 3),
    ("equities:Action_Monde", "R1.mu"): (41, 3),
    ("equities:Action_Monde", "R1.sigma"): (42, 3),
    ("equities:Action_Monde", "R2.mu"): (43, 3),
    ("equities:Action_Monde", "R2.sigma"): (44, 3),
    ("equities:Action_emergent", "R1.mu"): (50, 3),
    ("equities:Action_emergent", "R1.sigma"): (51, 3),
    ("equities:Action_emergent", "R2.mu"): (52, 3),
    ("equities:Action_emergent", "R2.sigma"): (53, 3),
}


def load_reference(path: str, sheet: str = "Parametres") -> dict:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    ref = {}
    for key, (r, c) in _REF_CELLS.items():
        try:
            ref[key] = float(raw.iat[r, c])
        except Exception:
            ref[key] = np.nan
    return ref


def _calibrated_value(cal, factor, param):
    if factor.startswith("equities:"):
        sheet = factor.split(":", 1)[1]
        regs = cal.regime.joint["regimes_by_equity"][sheet]   # régime COMMUN
        # La référence ne décrit que 2 régimes (R1 = stress, R2 = normal) ;
        # la chaîne commune peut en compter K* != 2. On compare les régimes
        # disponibles et on renvoie NaN au-delà (pas d'IndexError).
        if param.startswith("R1.") and len(regs) >= 1:
            return regs[0][param[3:]]
        if param.startswith("R2.") and len(regs) >= 2:
            return regs[1][param[3:]]
        return np.nan
    fr = cal.margins.get(factor)
    if fr is None:
        return np.nan
    return fr.params.get(param, np.nan)


def comparison_table(cal, ref_path: str) -> pd.DataFrame:
    ref = load_reference(ref_path)
    rows = []
    for (factor, param), refv in ref.items():
        myv = _calibrated_value(cal, factor, param)
        rel = (myv - refv) / refv * 100 if refv not in (0, np.nan) and not np.isnan(refv) else np.nan
        rows.append(dict(facteur=factor, param=param,
                         calibre=round(float(myv), 4) if myv == myv else np.nan,
                         reference=round(float(refv), 4) if refv == refv else np.nan,
                         ecart_pct=round(float(rel), 1) if rel == rel else np.nan))
    return pd.DataFrame(rows)


def v2f_method_comparison(cfg: dict, ref_path: str) -> pd.DataFrame:
    """Paramètres V2F sous les TROIS méthodologies (mle / ols / distributional)
    en regard de la référence, pour chaque facteur V2F de la config."""
    from .preprocessing import preprocess_factor
    from .margins import fit_v2f
    dt = float(cfg["data"]["dt"])
    ref = load_reference(ref_path)
    methods = ["mle", "ols", "distributional"]
    keys = ["kappa_short", "kappa_long", "sigma_short", "sigma_long", "mu", "rho_c"]
    rows = []
    for fac, spec in cfg["factors"].items():
        if spec.get("model") != "V2F":
            continue
        pre = preprocess_factor(fac, spec, cfg["data"])
        fits = {}
        for mth in methods:
            s = dict(spec); s["method"] = mth
            try:
                fits[mth] = fit_v2f(pre, s, dt).params
            except Exception:
                fits[mth] = {}
        for p in keys:
            row = dict(facteur=fac, param=p)
            for mth in methods:
                v = fits[mth].get(p, np.nan)
                row[mth] = round(float(v), 4) if v == v else np.nan
            rv = ref.get((fac, p), np.nan)
            row["reference"] = round(float(rv), 4) if rv == rv else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def pit_diagnostics(cal, alpha: float = 0.05) -> pd.DataFrame:
    """Adéquation des résidus PIT : moyenne, écart-type, KS (normalité), queue.

    Sous bonne spécification, chaque résidu standardisé est N(0,1) i.i.d. et la
    dépendance est une copule gaussienne. On teste la normalité (Kolmogorov–
    Smirnov) et on mesure l'excès de kurtosis (queue). Un rejet (p < alpha) ou
    un excès de kurtosis marqué signale un facteur où une copule/marge à queues
    (Student) serait préférable — cf. option `correlations.copula`.
    """
    series = {c: cal.margins[m].resid[c].dropna().values
              for m in cal.margins for c in cal.margins[m].resid.columns}
    # Groupe B : résidus standardisés (régime le plus probable a posteriori).
    if cal.regime is not None:
        J = cal.regime.joint
        rets, xi = J["returns"], J["xi"].values
        a_star = xi.argmax(axis=1)
        m_m, s_m = J["m_month"], J["s_month"]
        for j, nm in enumerate(rets.columns):
            x = rets[nm].values
            series[nm] = (x - m_m[a_star, j]) / s_m[a_star, j]
    rows = []
    for c, x in series.items():
        z = (x - x.mean()) / x.std(ddof=0)
        p = float(kstest(z, "norm").pvalue)
        rows.append(dict(composante=c, n=len(x), moyenne=round(float(x.mean()), 3),
                         ecart_type=round(float(x.std(ddof=0)), 3),
                         KS_p=round(p, 3),
                         kurtosis_exces=round(float(kurtosis(x)), 2),
                         normal=(p >= alpha)))
    return pd.DataFrame(rows)
