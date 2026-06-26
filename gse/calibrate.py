"""Orchestrateur de calibrage (cascade séquentielle IFM, cadre de la note).

Étape 1 : marges du Groupe A (EMV exact, formes fermées).
Étape 2 : marges du Groupe B (RSLN-2) + régime commun (EM).
Étape 3 : dépendance pleine par régime (sur résidus PIT), masque réglable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import yaml

from .preprocessing import preprocess_factor
from .margins import MARGIN_FITTERS, FitResult
from .regime import fit_rsln2, RegimeFitResult
from .dependence import compute_dependence

log = logging.getLogger("gse")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bs_vol_series(pre):
    """Série de rendements d'un facteur BS pour le régime commun : source de
    volatilité (déslissée si disponible, sinon brute), cohérente avec la
    convention BS (vol sur rendements déslissés)."""
    if (pre.meta.get("vol_source", "unsmoothed") == "unsmoothed"
            and "ret_unsmoothed" in pre.data):
        return pre.data["ret_unsmoothed"]
    return pre.data["ret_raw"]


@dataclass
class CalibrationResult:
    margins: dict = field(default_factory=dict)        # name -> FitResult (Gr.A)
    regime: RegimeFitResult | None = None              # Groupe B
    dependence: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def params_table(self) -> dict:
        out = {n: fr.params for n, fr in self.margins.items()}
        if self.regime is not None:
            J = self.regime.joint
            out["equities_joint"] = dict(
                K=int(J["K"]),
                P=J["P"].tolist(),
                pi=J["pi"].tolist(),
                regimes_by_equity=J["regimes_by_equity"])   # mu/sigma par actif et régime
            if self.regime.params_separate:                 # vide si compare_separate=false
                out["equities_separate"] = self.regime.params_separate
        return out


def calibrate(config) -> CalibrationResult:
    """Pipeline complet à partir d'un chemin de config ou d'un dict."""
    cfg = load_config(config) if isinstance(config, str) else config
    data_cfg = cfg["data"]
    dt = float(data_cfg["dt"])
    # rendre le chemin des données relatif au fichier de config si besoin
    base = os.path.dirname(config) if isinstance(config, str) else "."
    for key in ("path",):
        p = data_cfg[key]
        if not os.path.isabs(p) and not os.path.exists(p):
            cand = os.path.join(base, "..", p)
            if os.path.exists(cand):
                data_cfg[key] = cand

    # Prétraitement de tous les facteurs.
    pre_map = {}
    for name, spec in cfg["factors"].items():
        log.info("Prétraitement : %s (%s)", name, spec["model"])
        pre_map[name] = preprocess_factor(name, spec, data_cfg)

    # Facteurs BS *promus* au Groupe B (régime commun) : regime_sensitive_params.
    promoted = [name for name, spec in cfg["factors"].items()
                if spec["model"] == "BS" and spec.get("regime_sensitive_params", False)]

    margins, regime, rsln = {}, None, None
    for name, spec in cfg["factors"].items():
        model = spec["model"]
        if model == "RSLN2":
            rsln = (name, spec)
        elif name in promoted:
            log.info("Calibrage : %s (BS) promu au Groupe B (régime commun).", name)
        else:
            log.info("Calibrage : %s (%s, Groupe A).", name, model)
            margins[name] = MARGIN_FITTERS[model](pre_map[name], spec, dt)

    if rsln is not None:
        name, spec = rsln
        extra = {nm: _bs_vol_series(pre_map[nm]) for nm in promoted}
        regime = fit_rsln2(pre_map[name], spec, dt, extra_series=extra)
    elif promoted:
        raise ValueError("Facteurs BS promus au Groupe B mais aucun facteur "
                         "RSLN2 (chaîne commune) n'est déclaré.")

    dep = compute_dependence(list(margins.values()), regime, cfg)
    return CalibrationResult(margins=margins, regime=regime,
                             dependence=dep, config=cfg)
