"""GSE IRCANTEC — outil modulaire de prétraitement, calibrage et simulation.

Cadre de la note de recherche : représentation AR agrégée à régime latent
commun, calibrage par vraisemblance jointe (séquentiel, IFM), dépendance
pleinement estimée par régime, simulation cohérente.
"""
from .calibrate import calibrate, load_config, CalibrationResult  # noqa: F401

__all__ = ["calibrate", "load_config", "CalibrationResult"]
__version__ = "0.1.0"
