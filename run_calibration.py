#!/usr/bin/env python3
"""Pilote de bout en bout : prétraitement -> calibrage -> comparaison -> simulation.

Usage :
    python run_calibration.py [--config gse/config.yaml] [--paths 2000]
                              [--no-sim] [--out outputs]
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from gse.calibrate import calibrate, load_config
from gse.compare import comparison_table
from gse.simulate import simulate, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gse/config.yaml")
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 120)
    os.makedirs(args.out, exist_ok=True)

    cfg = load_config(args.config)
    print("\n" + "=" * 78 + "\n CALIBRAGE\n" + "=" * 78)
    cal = calibrate(cfg)

    # --- paramètres calibrés ---
    params = cal.params_table()
    with open(os.path.join(args.out, "parametres_calibres.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=float)
    print("\nParamètres (marges Groupe A) :")
    for nm, fr in cal.margins.items():
        pretty = {k: round(v, 4) for k, v in fr.params.items()
                  if isinstance(v, (int, float))}
        print(f"  [{fr.model:4}] {nm:14}", pretty)
    if cal.regime is not None:
        print("\nActions (chaînes séparées, réf.) :")
        for nm, sp in cal.regime.params_separate.items():
            r1, r2 = sp["regimes"][0], sp["regimes"][1]
            print(f"  {nm:16} R1(mu/sig)={r1['mu']:.2f}/{r1['sigma']:.2f}  "
                  f"R2={r2['mu']:.2f}/{r2['sigma']:.2f}  "
                  f"p1->2={sp['p_1to2']:.4f} p2->1={sp['p_2to1']:.4f}")

    # --- comparaison à la référence ---
    print("\n" + "=" * 78 + "\n COMPARAISON À Parametres_models.xlsx (colonne 2026)\n" + "=" * 78)
    ref_cfg = cfg.get("reference", {})
    ref_path = ref_cfg.get("path", "Parametres_models.xlsx")
    if not os.path.exists(ref_path):
        cand = os.path.join(os.path.dirname(args.config), "..", ref_path)
        ref_path = cand if os.path.exists(cand) else ref_path
    try:
        cmp = comparison_table(cal, ref_path)
        cmp.to_csv(os.path.join(args.out, "comparaison_parametres.csv"), index=False)
        print(cmp.to_string(index=False))
        ok = cmp.dropna(subset=["ecart_pct"])
        print(f"\n|écart| médian = {ok['ecart_pct'].abs().median():.1f}%   "
              f"|écart|<10% : {(ok['ecart_pct'].abs()<10).mean()*100:.0f}% des paramètres")
    except Exception as e:
        print("Comparaison indisponible :", e)

    # --- corrélations par régime ---
    dep = cal.dependence
    for lbl, Om in dep["regimes"].items():
        Om.to_csv(os.path.join(args.out, f"correlation_{lbl}.csv"))
    print(f"\nMatrices de corrélation par régime : {list(dep['regimes'])} "
          f"(masque={dep.get('mask','-')}, d={len(dep['components'])})")

    # --- simulation ---
    if not args.no_sim:
        print("\n" + "=" * 78 + "\n SIMULATION\n" + "=" * 78)
        sim = simulate(cal, n_paths=args.paths, horizon_years=args.horizon)
        summ = summarize(sim, dt=float(cfg["data"]["dt"]))
        print(summ.to_string(index=False))
        np.save(os.path.join(args.out, "regime_paths.npy"), sim["_regime"])
        print(f"\nSorties simulées : {len([k for k in sim if not k.startswith('_')])} séries, "
              f"{sim['_regime'].shape[0]} trajectoires x {sim['_regime'].shape[1]-1} pas.")

    print("\nTerminé. Sorties dans :", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
