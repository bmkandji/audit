"""Prétraitement des données du GSE.

Fournit le chargement des séries depuis le classeur Excel, la sélection de la
fenêtre de calibrage, et les transformations par facteur (rendements
100*log, log/log100, déslissage AR(1), ajout de rendement locatif).

Toutes les fonctions sont pures et validées : elles lèvent des erreurs
explicites en cas de données manquantes ou de configuration incohérente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Chargement et fenêtrage
# --------------------------------------------------------------------------- #
def load_sheet(path: str, sheet: str) -> pd.Series:
    """Charge une feuille (col. 0 = Date, col. 1 = valeur) en série datée."""
    df = pd.read_excel(path, sheet_name=sheet)
    if df.shape[1] < 2:
        raise ValueError(f"Feuille '{sheet}' : au moins 2 colonnes attendues.")
    df = df.iloc[:, :2].copy()
    df.columns = ["Date", "value"]
    df["Date"] = pd.to_datetime(df["Date"])
    s = df.dropna().set_index("Date")["value"].astype(float).sort_index()
    if s.empty:
        raise ValueError(f"Feuille '{sheet}' : série vide après nettoyage.")
    return s


def apply_window(s: pd.Series, start=None, end=None) -> pd.Series:
    """Restreint la série à [start, end] (bornes incluses, None = ouvert)."""
    if start is not None:
        s = s[s.index >= pd.to_datetime(start)]
    if end is not None:
        s = s[s.index <= pd.to_datetime(end)]
    if len(s) < 3:
        raise ValueError("Fenêtre de calibrage trop courte (< 3 points).")
    return s


# --------------------------------------------------------------------------- #
#  Transformations élémentaires
# --------------------------------------------------------------------------- #
def log_returns_100(prices: pd.Series) -> pd.Series:
    """100 * log(P_t / P_{t-1}) — pour les indices."""
    if (prices <= 0).any():
        raise ValueError("Rendements log : l'indice doit être strictement positif.")
    return 100.0 * np.log(prices / prices.shift(1)).dropna()


def to_log(s: pd.Series) -> pd.Series:
    if (s <= 0).any():
        raise ValueError("log : la série doit être strictement positive.")
    return np.log(s)


def to_log100(s: pd.Series) -> pd.Series:
    return 100.0 * to_log(s)


def unsmooth_ar1(returns: pd.Series) -> tuple[pd.Series, float]:
    """Déslissage AR(1) (Lizieri et al. 2012).

    r*_t = (r_t - b r_{t-1}) / (1 - b), où b est le coefficient AR(1) estimé
    par MCO sur r_t = a + b r_{t-1} + e. Retourne (série déslissée, b).
    """
    r = returns.dropna().values
    if len(r) < 4:
        raise ValueError("Déslissage : série trop courte.")
    r0, r1 = r[:-1], r[1:]
    b = float(np.polyfit(r0, r1, 1)[0])
    if abs(1.0 - b) < 1e-8:
        raise ValueError("Déslissage : b trop proche de 1 (instable).")
    r_star = (r1 - b * r0) / (1.0 - b)
    idx = returns.dropna().index[1:]
    return pd.Series(r_star, index=idx, name=returns.name), b


# --------------------------------------------------------------------------- #
#  Résultat de prétraitement par facteur
# --------------------------------------------------------------------------- #
@dataclass
class Preprocessed:
    """Conteneur des séries prêtes au calibrage pour un facteur."""
    name: str
    model: str
    # série(s) modélisée(s), alignée(s) sur un index commun
    data: dict[str, pd.Series]
    meta: dict = field(default_factory=dict)

    @property
    def index(self) -> pd.DatetimeIndex:
        any_series = next(iter(self.data.values()))
        return any_series.index


# --------------------------------------------------------------------------- #
#  Aiguillage par facteur (piloté par la config)
# --------------------------------------------------------------------------- #
def preprocess_factor(name: str, spec: dict, data_cfg: dict) -> Preprocessed:
    """Applique le prétraitement déclaré dans la config pour un facteur."""
    path = data_cfg["path"]
    win = data_cfg.get("window", {}) or {}
    start, end = win.get("start"), win.get("end")
    model = spec["model"]
    pp = spec.get("preprocessing", {}) or {}
    transform = pp.get("transform", "none")

    def _load(sheet):
        return apply_window(load_sheet(path, sheet), start, end)

    # ---- V2F : deux séries (court, long) en niveau de taux --------------
    if model == "V2F":
        qs = _load(spec["sheets"]["short"])
        ql = _load(spec["sheets"]["long"])
        if transform in ("log", "log100"):
            qs, ql = (to_log100(qs), to_log100(ql)) if transform == "log100" else (to_log(qs), to_log(ql))
        idx = qs.index.intersection(ql.index)
        if len(idx) < 12:
            raise ValueError(f"{name}: recouvrement court/long insuffisant.")
        return Preprocessed(name, model, {"short": qs.loc[idx], "long": ql.loc[idx]},
                            meta={"transform": transform})

    # ---- CIR : un spread positif ---------------------------------------
    if model == "CIR":
        s = _load(spec["sheet"])
        if transform in ("log", "log100"):
            raise ValueError("CIR : ne pas transformer en log (modèle sur le spread).")
        if (s <= 0).any():
            raise ValueError(f"{name}: spread CIR non strictement positif.")
        return Preprocessed(name, model, {"level": s}, meta={"transform": "none"})

    # ---- BK : OU sur la série de log-spread -----------------------------
    if model == "BK":
        s = _load(spec["sheet"])
        if transform == "log100":
            s = to_log100(s)
        elif transform == "log":
            s = to_log(s)
        elif transform != "none":
            raise ValueError(f"{name}: transform BK invalide '{transform}'.")
        # transform 'none' : la série est supposée déjà en (100*)log-spread
        return Preprocessed(name, model, {"y": s}, meta={"transform": transform})

    # ---- BS : log-rendements (déslissés) d'un indice --------------------
    if model == "BS":
        px = _load(spec["sheet"])
        if transform != "log_return_100":
            raise ValueError(f"{name}: BS requiert transform=log_return_100.")
        r_raw = log_returns_100(px)
        out = {"ret_raw": r_raw}
        meta = {"transform": transform,
                "mean_source": pp.get("mean_source", "raw"),
                "vol_source": pp.get("vol_source", "unsmoothed"),
                "income_yield": float(pp.get("income_yield", 0.0))}
        if pp.get("unsmooth", False):
            r_star, b = unsmooth_ar1(r_raw)
            out["ret_unsmoothed"] = r_star
            meta["ar1_b"] = b
        return Preprocessed(name, model, out, meta=meta)

    # ---- RSLN2 : log-rendements de plusieurs indices --------------------
    if model == "RSLN2":
        if transform != "log_return_100":
            raise ValueError(f"{name}: RSLN2 requiert transform=log_return_100.")
        data = {}
        for sheet in spec["sheets"]:
            data[sheet] = log_returns_100(_load(sheet))
        return Preprocessed(name, model, data,
                            meta={"transform": transform,
                                  "common_regime": spec.get("common_regime", False),
                                  "n_states": spec.get("n_states", 2),
                                  "em_restarts": spec.get("em_restarts", 12),
                                  "var_floor": spec.get("var_floor", 1e-6)})

    raise ValueError(f"{name}: modèle inconnu '{model}'.")
