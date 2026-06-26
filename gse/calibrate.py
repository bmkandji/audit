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


@dataclass
class CalibrationResult:
    margins: dict = field(default_factory=dict)        # name -> FitResult (Gr.A)
    regime: RegimeFitResult | None = None              # Groupe B
    dependence: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def params_table(self) -> dict:
        out = {n: fr.params for n, fr in self.margins.items()}
        if self.regime is not None:
            out["equities_separate"] = self.regime.params_separate
            out["equities_joint"] = dict(
                P=self.regime.joint["P"].tolist(),
                pi=self.regime.joint["pi"].tolist())
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

    margins, regime = {}, None
    for name, spec in cfg["factors"].items():
        model = spec["model"]
        log.info("Prétraitement + calibrage : %s (%s)", name, model)
        pre = preprocess_factor(name, spec, data_cfg)
        if model == "RSLN2":
            regime = fit_rsln2(pre, spec, dt)
        else:
            fitter = MARGIN_FITTERS[model]
            margins[name] = fitter(pre, spec, dt)

    dep = compute_dependence(list(margins.values()), regime, cfg)
    return CalibrationResult(margins=margins, regime=regime,
                             dependence=dep, config=cfg)
