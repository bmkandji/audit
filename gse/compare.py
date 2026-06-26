"""Comparaison des paramètres calibrés avec la référence Parametres_models.xlsx."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
