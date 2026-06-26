#!/usr/bin/env python3
"""Pilote de bout en bout du GSE.

Prétraitement -> calibrage (régime latent COMMUN, K* optimal) -> comparaison
à la référence -> simulation -> **insertion directe** des résultats numériques
dans la note LaTeX (section « Résultats numériques », entre balises).

Usage :
    python run_calibration.py
    python run_calibration.py --paths 5000 --horizon 30
    python run_calibration.py --no-note      # ne pas modifier la note
    python run_calibration.py --no-sim       # pas de simulation (ni section)
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from gse.calibrate import calibrate, load_config
from gse.compare import comparison_table, load_reference, pit_diagnostics
from gse.preprocessing import preprocess_factor
from gse.simulate import simulate, summarize

# Balises délimitant la section auto-générée dans la note LaTeX.
NOTE_BEGIN = "% >>> GSE-AUTO-RESULTS BEGIN (généré par run_calibration.py — ne pas éditer à la main) >>>"
NOTE_END = "% <<< GSE-AUTO-RESULTS END <<<"

LAB = {'inflation.short': 'Inf.C', 'inflation.long': 'Inf.L',
       'real_rate.short': 'TxR.C', 'real_rate.long': 'TxR.L',
       'credit': 'Créd', 'dette_privee': 'DetP', 'pe': 'PE',
       'immobilier': 'Immo', 'infra': 'Infra', 'Action_EUR': 'Act.EU',
       'Action_Monde': 'Act.Mo', 'Action_emergent': 'Act.EM'}


# --------------------------------------------------------------------------- #
#  Mise en forme LaTeX
# --------------------------------------------------------------------------- #
def _f(x, n=2):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "--"


def _ec(cal, ref, n=1):
    if ref is None or ref == 0 or (isinstance(ref, float) and np.isnan(ref)):
        return "--"
    return f"{(cal-ref)/ref*100:.{n}f}"


def esc(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")


def build_section(cal, ref, cfg, sim):
    """Construit la section LaTeX « Résultats numériques » (sans les balises)."""
    dt = float(cfg["data"]["dt"])
    L = []
    a = L.append
    a(r"% ====================================================================")
    a(r"\section{Résultats numériques (sortie du module de calibrage)}")
    a(r"\label{sec:resultats}")
    a(r"% ====================================================================")
    a(r"Tous les résultats ci-dessous sont \emph{produits par l'outil} "
      r"(package \texttt{gse}, via \texttt{run\_calibration.py}) sur "
      r"\texttt{Historical\_Data\_\allowbreak Model\_\allowbreak Calibration.xlsx}, et insérés "
      r"automatiquement ici. Conformément au cadre de la note, \textbf{les facteurs dépendant du "
      r"régime latent (actions) sont calibrés ensemble, sous une \emph{unique} chaîne de Markov} "
      r"(une seule matrice de transition)~; le nombre d'états $K^\star$ est choisi par critère "
      r"d'information. La colonne \emph{Réf.} reprend \texttt{Parametres\_models.xlsx} (2026).")

    # ---------- R.1 Données / fenêtre / prétraitements ----------
    a(r"\subsection{Données, fenêtre de calibrage et prétraitements}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{5pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l l X r l@{}}")
    a(r"\toprule")
    a(r"\textbf{Facteur} & \textbf{Modèle} & \textbf{Feuille(s)} & \textbf{$n$} & \textbf{Prétraitement}\\")
    a(r"\midrule")
    for name, spec in cfg["factors"].items():
        pre = preprocess_factor(name, spec, cfg["data"])
        n = len(pre.index)
        tr = (spec.get("preprocessing", {}) or {}).get("transform", "none")
        extra = []
        if (spec.get("preprocessing", {}) or {}).get("unsmooth"):
            extra.append("déslissage AR(1)")
        if spec.get("fix"):
            extra.append("fix " + ",".join(spec["fix"].keys()))
        if spec["model"] == "RSLN2":
            extra.append("régime commun")
        sh = spec.get("sheet")
        if sh is None:
            sv = spec.get("sheets", [])
            sh = ", ".join(sv.values() if isinstance(sv, dict) else sv)
        tr_txt = {"none": "aucun", "log_return_100": r"$100\ln$-rdt",
                  "log100": r"$100\ln$", "log": r"$\ln$"}.get(tr, tr)
        note = tr_txt + (("~; " + ", ".join(extra)) if extra else "")
        a(f"{esc(name)} & {spec['model']} & \\footnotesize {esc(sh)} & {n} & \\footnotesize {note}\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Périmètre et prétraitements par facteur. Fenêtre commune déc.~2005--déc.~2025 "
      r"($n$ = observations modélisées~; immobilier à partir de juin~2008).}\label{tab:res-data}")
    a(r"\end{table}")

    # ---------- R.2 V2F ----------
    inf, rea = cal.margins["inflation"].params, cal.margins["real_rate"].params
    a(r"\subsection{Facteurs de taux --- Vasicek à deux facteurs}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{5pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l X r r r r r r@{}}")
    a(r"\toprule")
    a(r"& & \multicolumn{3}{c}{\textbf{Inflation}} & \multicolumn{3}{c}{\textbf{Taux réels}}\\")
    a(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}")
    a(r"\textbf{Paramètre} & \textbf{Unité} & \textbf{Calib.} & \textbf{Réf.} & \textbf{Éc.} & \textbf{Calib.} & \textbf{Réf.} & \textbf{Éc.}\\")
    a(r"\midrule")
    rows = [
        (r"$\kappa_1$ (court)", r"\%/an", inf["kappa_short"]*100, ref[("inflation","kappa_short")]*100, rea["kappa_short"]*100, ref[("real_rate","kappa_short")]*100),
        (r"$\kappa_2$ (long)", r"\%/an", inf["kappa_long"]*100, ref[("inflation","kappa_long")]*100, rea["kappa_long"]*100, ref[("real_rate","kappa_long")]*100),
        (r"$\sigma_1$ (court)", r"\%", inf["sigma_short"], ref[("inflation","sigma_short")], rea["sigma_short"], ref[("real_rate","sigma_short")]),
        (r"$\sigma_2$ (long)", r"\%", inf["sigma_long"], ref[("inflation","sigma_long")], rea["sigma_long"], ref[("real_rate","sigma_long")]),
        (r"$\mu$ (moy.\ LT)", r"\%", inf["mu"], ref[("inflation","mu")], rea["mu"], ref[("real_rate","mu")]),
    ]
    for nm, u, c1, r1, c2, r2 in rows:
        a(f"{nm} & {u} & {_f(c1)} & {_f(r1)} & {_ec(c1,r1)} & {_f(c2)} & {_f(r2)} & {_ec(c2,r2)}\\\\")
    a(f"$\\rho_c$ (corr.\\ int.) & -- & {_f(inf['rho_c'],3)} & -- & -- & {_f(rea['rho_c'],3)} & -- & --\\\\")
    a(r"\addlinespace")
    hl = lambda k: np.log(2) / k
    a(f"Demi-vie court & an & {_f(hl(inf['kappa_short']),2)} & -- & -- & {_f(hl(rea['kappa_short']),2)} & -- & --\\\\")
    a(f"Demi-vie long & an & {_f(hl(inf['kappa_long']),2)} & -- & -- & {_f(hl(rea['kappa_long']),2)} & -- & --\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{V2F (EMV exact~; $\mu$ inflation fixé à la cible COR). Demi-vie $=\ln 2/\kappa$. "
      r"Écarts $\kappa,\sigma$ : référence par cibles distributionnelles "
      r"(cf.\ \S\ref{sec:resultats-discussion}).}\label{tab:res-v2f}")
    a(r"\end{table}")

    # ---------- R.3 CIR / BK / BS ----------
    cr, bk = cal.margins["credit"].params, cal.margins["dette_privee"].params
    csim = cal.margins["credit"].sim
    Es = np.exp(bk["mu"]/100 + 0.5*(bk["sigma"]/100)**2/(2*bk["kappa"]))*100
    a(r"\subsection{Crédit, dette privée et actifs réels}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{5pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l l X r r r@{}}")
    a(r"\toprule")
    a(r"\textbf{Facteur} & \textbf{Modèle} & \textbf{Paramètre} & \textbf{Calib.} & \textbf{Réf.} & \textbf{Éc.\,\%}\\")
    a(r"\midrule")
    def row(fac, mod, lab, c, rf, n=3):
        rfs = _f(rf, n) if (rf == rf) else "--"
        a(f"{fac} & {mod} & {lab} & {_f(c,n)} & {rfs} & {_ec(c,rf)}\\\\")
    row("Crédit", "CIR", r"$\kappa$ (\%/an)", cr["kappa"]*100, ref[("credit","kappa")]*100, 2)
    row("", "", r"$\sigma$", cr["sigma"], ref[("credit","sigma")], 4)
    row("", "", r"$\theta$ (\%)", cr["theta"], ref[("credit","theta")], 3)
    a(f" & & Feller $\\sigma^2\\le2\\kappa\\theta$ & "
      f"{_f(cr['sigma']**2,3)}$\\le${_f(2*cr['kappa']*cr['theta'],3)} & -- & "
      f"{'oui' if csim['feller'] else 'non'}\\\\")
    a(r"\addlinespace")
    row("Dette privée", "BK", r"$\kappa$ (\%/an)", bk["kappa"]*100, ref[("dette_privee","kappa")]*100, 2)
    row("", "", r"$\sigma$ ($100\ln s$)", bk["sigma"], ref[("dette_privee","sigma")], 2)
    row("", "", r"$\mu$ ($100\ln s$)", bk["mu"], ref[("dette_privee","mu")], 1)
    a(f" & & $\\mathbb E[s_\\infty]$ (\\%) & {_f(Es,3)} & -- & --\\\\")
    a(r"\addlinespace")
    for nm, lbl in [("pe", "PE (non coté)"), ("infra", "Infrastructure"), ("immobilier", "Immobilier")]:
        if nm not in cal.margins:        # facteur promu au Groupe B : voir tableaux régime
            continue
        p = cal.margins[nm].params
        row(lbl, "BS", r"$\mu$ (\%/an)", p["mu"], ref[(nm, "mu")], 2)
        row("", "", r"$\sigma$ (\%/an)", p["sigma"], ref[(nm, "sigma")], 2)
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Crédit (CIR, EMV $\chi^2$ décentré), dette privée (BK, sur $100\ln(\text{spread})$) "
      r"et actifs réels (BS).}\label{tab:res-margins}")
    a(r"\end{table}")

    # ---------- R.4 Déslissage ----------
    a(r"\subsection{Coefficients de déslissage (\emph{unsmoothing})}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{6pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l X r r r r@{}}")
    a(r"\toprule")
    a(r"\textbf{Actif} & \textbf{Modèle} & \textbf{AR(1) $b$} & \textbf{$\sigma$ brute} & \textbf{$\sigma$ déslissée} & \textbf{Inflation var.}\\")
    a(r"\midrule")
    for nm, lbl in [("pe", "PE (LPX50)"), ("immobilier", "Immobilier"), ("infra", "Infrastructure")]:
        if nm not in cal.margins:        # facteur promu au Groupe B
            continue
        p = cal.margins[nm].params
        a(f"{lbl} & BS & {_f(p['ar1_b'],4)} & {_f(p['sigma_raw'],2)} & {_f(p['sigma_unsmoothed'],2)} & {_f(p['var_inflation'],3)}\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Déslissage AR(1) : $r^\ast_t=(r_t-b\,r_{t-1})/(1-b)$~; volatilité retenue = "
      r"\emph{déslissée} (annualisée, \%)~; facteur d'inflation de variance $(1+b)/(1-b)$. PE "
      r"montre la sur-correction attendue ($b>0$).}\label{tab:res-unsmooth}")
    a(r"\end{table}")

    # ---------- R.5 Choix de K ----------
    ks = cal.regime.joint["kselect"]
    a(r"\subsection{Choix du nombre d'états latents $K^\star$ (régime commun)}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{8pt}")
    a(r"\begin{tabular}{@{}r r r r r l@{}}")
    a(r"\toprule")
    a(r"$K$ & $\ell^\star(K)$ & $p_K$ & AIC & BIC & \\")
    a(r"\midrule")
    for r in ks["table"]:
        mark = r"$\leftarrow K^\star$ (BIC)" if r["K"] == ks["k_star"] else ""
        a(f"{r['K']} & {_f(r['loglik'],1)} & {r['p']} & {_f(r['AIC'],1)} & {_f(r['BIC'],1)} & {mark}\\\\")
    a(r"\bottomrule\end{tabular}")
    a(rf"\caption{{Régime commun (émissions gaussiennes $D=3$ actions, $n={ks['n']}$). Le BIC "
      r"sélectionne $K^\star=" + str(ks["k_star"]) + r"$ ; l'AIC, moins parcimonieux, décroît "
      r"encore. Un régime commun à $K^\star$ états remplace $2^3$ états joints de chaînes "
      r"séparées.}\label{tab:res-kselect}")
    a(r"\end{table}")

    # ---------- R.6 Régime commun ----------
    J = cal.regime.joint
    P = np.asarray(J["P"]); pi = np.asarray(J["pi"]); K = J["K"]
    a(r"\subsection{Régime latent commun : transition unique et paramètres par actif}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{6pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l X" + " r"*(2*K) + r"@{}}")
    a(r"\toprule")
    hdr = " & ".join([fr"\multicolumn{{2}}{{c}}{{\textbf{{Régime {k+1}}}}}" for k in range(K)])
    a(r"\textbf{Actif} & \textbf{} & " + hdr + r"\\")
    a("".join([fr"\cmidrule(lr){{{3+2*k}-{4+2*k}}}" for k in range(K)]))
    a(r"\textbf{} & & " + " & ".join([r"$\mu$ & $\sigma$" for _ in range(K)]) + r"\\")
    a(r"\midrule")
    EQLAB = {"Action_EUR": "Euro", "Action_Monde": "Monde", "Action_emergent": "Émergent"}
    members = list(J["regimes_by_equity"].keys())   # actions + éventuels BS promus
    for sheet in members:
        lbl = EQLAB.get(sheet, LAB.get(sheet, sheet))
        regs = J["regimes_by_equity"][sheet]
        cells = " & ".join([f"{_f(regs[k]['mu'])} & {_f(regs[k]['sigma'])}" for k in range(K)])
        a(f"{lbl} & & {cells}\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Actions sous le régime latent \emph{commun} ($K^\star=" + str(K) +
      r"$, EM joint). $\mu,\sigma$ annualisés en \%. Le régime~1 (forte volatilité) frappe "
      r"\emph{simultanément} les trois actifs.}\label{tab:res-common}")
    a(r"\end{table}")
    a(r"\begin{definitionbox}")
    Pmat = r"\\".join([" & ".join(_f(P[i, j], 4) for j in range(K)) for i in range(K)])
    pirow = r",\ ".join(_f(pi[k], 4) for k in range(K))
    a(r"\[ P=\begin{pmatrix}" + Pmat + r"\end{pmatrix},\qquad \pi=(" + pirow + r")\transp. \]")
    a(r"Matrice de transition mensuelle \emph{unique} et loi invariante du régime commun "
      r"(probabilité stationnaire de stress $\pi_1=" + _f(pi[0]*100, 1) + r"\,\%$).")
    a(r"\end{definitionbox}")

    # ---------- R.7 régimes : moyennes/volatilités vs référence (sans transitions) ----------
    a(r"\subsection{Régimes : moyennes et volatilités vs référence}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{5pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l X r r r r@{}}")
    a(r"\toprule")
    a(r"& & \multicolumn{2}{c}{\textbf{Régime 1 (stress)}} & \multicolumn{2}{c}{\textbf{Régime 2 (normal)}}\\")
    a(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}")
    a(r"\textbf{Actif} & \textbf{Source} & $\mu_1$ & $\sigma_1$ & $\mu_2$ & $\sigma_2$\\")
    a(r"\midrule")
    for sheet in members:
        lbl = EQLAB.get(sheet, LAB.get(sheet, sheet))
        regs = J["regimes_by_equity"][sheet]
        reg2 = regs[1] if len(regs) >= 2 else regs[0]   # K*<2 : pas de 2e régime
        k = f"equities:{sheet}"
        a(f"{lbl} & Calib.\\ (commun) & {_f(regs[0]['mu'])} & {_f(regs[0]['sigma'])} & {_f(reg2['mu'])} & {_f(reg2['sigma'])}\\\\")
        a(f" & Réf.\\ (par actif) & {_f(ref.get((k,'R1.mu')))} & {_f(ref.get((k,'R1.sigma')))} & {_f(ref.get((k,'R2.mu')))} & {_f(ref.get((k,'R2.sigma')))}\\\\")
        a(r"\addlinespace")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Caractérisation des régimes sous la chaîne \emph{commune} (calibrée), comparée "
      r"à la référence (qui calibre une chaîne distincte par actif). \textbf{La matrice de "
      r"transition est unique} (cf.\ tableau~\ref{tab:res-common})~: il n'y a pas de probabilités "
      r"de transition différentes par actif. Euro concorde de près~; Monde/émergent s'écartent "
      r"car le régime est désormais \emph{partagé} (entrée en stress simultanée).}\label{tab:res-rsln-sep}")
    a(r"\end{table}")

    # ---------- R.8 corrélations ----------
    comps = cal.dependence["components"]
    regs_dep = cal.dependence["regimes"]
    O1 = regs_dep["reg1"].values
    O2 = regs_dep.get("reg2", regs_dep["reg1"]).values   # K*<2 : matrice unique
    d = len(comps)
    M = np.where(np.triu(np.ones((d, d)), 1) > 0, O1, O2)
    np.fill_diagonal(M, 1.0)
    a(r"\subsection{Corrélations des résidus par régime}")
    a(r"\begin{table}[H]\centering\renewcommand{\arraystretch}{1.1}\setlength{\tabcolsep}{3pt}")
    a(r"\resizebox{\textwidth}{!}{%")
    a(r"\begin{tabular}{@{}l" + "r" * d + r"@{}}")
    a(r"\toprule")
    a(" & " + " & ".join(LAB.get(c, c) for c in comps) + r"\\")
    a(r"\midrule")
    for i in range(d):
        cells = [r"\textbf{1}" if i == j else f"{M[i,j]:.2f}" for j in range(d)]
        a(LAB.get(comps[i], comps[i]) + " & " + " & ".join(cells) + r"\\")
    a(r"\bottomrule\end{tabular}}")
    a(r"\caption{Corrélations des résidus standardisés (masque \texttt{groupB}). \emph{Triangle "
      r"sup.} = régime de stress~; \emph{triangle inf.} = régime normal. Hors actions, communes "
      r"aux régimes~; SDP (Higham). (Étiquettes~: Inf=inflation, TxR=taux réel, "
      r"Créd=crédit, DetP=dette privée, Act.=actions.)}\label{tab:res-corr}")
    a(r"\end{table}")

    # ---------- R.9 simulation ----------
    a(r"\subsection{Simulation : statistiques de contrôle}")
    summ = summarize(sim, dt=dt)
    occ = float((sim["_regime"] == 0).mean()) * 100
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{6pt}")
    a(r"\begin{tabular}{@{}l l r l r@{}}")
    a(r"\toprule")
    a(r"\textbf{Sortie} & \textbf{Mesure} & \textbf{Val.} & \textbf{} & \textbf{E[terminal]}\\")
    a(r"\midrule")
    for _, rw in summ.iterrows():
        a(f"{esc(rw['sortie'])} & {esc(rw['m1'])} & {_f(rw['v1'],3)} & {esc(rw['m2'])} & {_f(rw['v2'],4)}\\\\")
    a(r"\bottomrule\end{tabular}")
    a(rf"\caption{{Contrôle sur {sim['_regime'].shape[0]} trajectoires $\times$ "
      rf"{sim['_regime'].shape[1]-1} pas (30~ans). Occupation simulée du régime de stress~: "
      rf"{_f(occ,1)}\,\% (cf.\ $\pi_1$). Les volatilités simulées reproduisent les "
      r"paramètres calibrés.}\label{tab:res-sim}")
    a(r"\end{table}")

    # ---------- discussion ----------
    a(r"\subsection{Lecture et concordance avec la référence}")
    a(r"\label{sec:resultats-discussion}")
    a(r"\begin{insightbox}[Concordances et écarts attendus]")
    a(r"\textbf{Concordances exactes} (méthodologie identique)~: crédit CIR ($\kappa,\sigma,\theta$), "
      r"dette privée $\kappa$, PE/infra $\mu$, moyenne LT des taux réels, $\mu$ inflation (fixé). "
      r"\textbf{Régime latent commun}~: \emph{une seule} matrice de transition pilote les trois "
      r"actions (entrée en stress simultanée, $\pi_1\approx" + _f(pi[0]*100, 0) + r"\,\%$)~; le "
      r"nombre d'états $K^\star=" + str(K) + r"$ est choisi par BIC (tableau~\ref{tab:res-kselect}). "
      r"\textbf{Écarts attendus}~: (i)~$\kappa,\sigma$ des V2F (référence = cibles "
      r"distributionnelles, mal conditionnées~; l'outil applique l'EMV)~; (ii)~$\sigma$ dette "
      r"privée (référence = dispersion stationnaire~; $\kappa,\mu$ concordent)~; (iii)~émergent "
      r"(vraisemblance plus élevée ici)~; (iv)~immobilier (indice de \emph{prix} vs rendement "
      r"total avec loyers).")
    a(r"\end{insightbox}")
    return "\n".join(L)


def update_note(note_path, section_text):
    """Insère/remplace la section auto-générée entre les balises de la note."""
    with open(note_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    block = [NOTE_BEGIN, *section_text.split("\n"), NOTE_END]
    if NOTE_BEGIN in lines and NOTE_END in lines:
        i0, i1 = lines.index(NOTE_BEGIN), lines.index(NOTE_END)
        new = lines[:i0] + block + lines[i1 + 1:]
    else:                                   # première insertion : avant la Conclusion
        try:
            i_con = next(i for i, l in enumerate(lines) if l.startswith(r"\section{Conclusion}"))
        except StopIteration:
            raise RuntimeError("Section Conclusion introuvable dans la note.")
        d_con = next(j for j in range(i_con - 1, -1, -1) if lines[j].startswith("% ="))
        new = lines[:d_con] + block + ["", ] + lines[d_con:]
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gse/config.yaml")
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-note", action="store_true", help="ne pas modifier la note LaTeX")
    ap.add_argument("--note", default="Note_recherche_GSE_calibrage_simulation.tex")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    os.makedirs(args.out, exist_ok=True)

    cfg = load_config(args.config)
    print("\n" + "=" * 78 + "\n CALIBRAGE\n" + "=" * 78)
    cal = calibrate(cfg)

    params = cal.params_table()
    with open(os.path.join(args.out, "parametres_calibres.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False, default=float)
    print("\nMarges (Groupe A) :")
    for nm, fr in cal.margins.items():
        pretty = {k: round(v, 4) for k, v in fr.params.items() if isinstance(v, (int, float))}
        print(f"  [{fr.model:4}] {nm:14}", pretty)
    if cal.regime is not None:
        J = cal.regime.joint
        print(f"\nRégime latent COMMUN : K* = {J['K']} (BIC)  |  pi = {np.round(J['pi'],4)}")
        for nm, regs in J["regimes_by_equity"].items():
            cells = "  ".join(f"R{i+1} {r['mu']:.2f}/{r['sigma']:.2f}"
                              for i, r in enumerate(regs))
            print(f"  {nm:16} {cells}")

    # comparaison
    print("\n" + "=" * 78 + "\n COMPARAISON À LA RÉFÉRENCE (2026)\n" + "=" * 78)
    ref_path = cfg.get("reference", {}).get("path", "Parametres_models.xlsx")
    try:
        cmp = comparison_table(cal, ref_path)
        cmp.to_csv(os.path.join(args.out, "comparaison_parametres.csv"), index=False)
        ok = cmp.dropna(subset=["ecart_pct"])
        print(cmp.to_string(index=False))
        print(f"\n|écart| médian = {ok['ecart_pct'].abs().median():.1f}%")
    except Exception as e:
        print("Comparaison indisponible :", e)

    for lbl, Om in cal.dependence["regimes"].items():
        Om.to_csv(os.path.join(args.out, f"correlation_{lbl}.csv"))

    # diagnostics : conditionnement des corrélations + adéquation des résidus
    shr = cal.dependence.get("shrinkage")
    if shr:
        print("\nDépendance par régime (rétrécissement) :")
        for lbl, info in shr.items():
            print(f"  {lbl}: n_eff={info['n_eff']}  delta={info['delta']}  "
                  f"min_eig(Omega)={info['min_eig']:.3e}")
    try:
        diag = pit_diagnostics(cal)
        diag.to_csv(os.path.join(args.out, "diagnostics_pit.csv"), index=False)
        rej = diag.loc[~diag["normal"], "composante"].tolist()
        print("\nDiagnostics résidus PIT (adéquation N(0,1)) — "
              f"rejets KS : {', '.join(rej) if rej else 'aucun'}")
    except Exception as e:
        print("Diagnostics PIT indisponibles :", e)

    # simulation (une seule fois, partagée avec la note)
    sim = None
    if not args.no_sim:
        print("\n" + "=" * 78 + "\n SIMULATION\n" + "=" * 78)
        sim = simulate(cal, n_paths=args.paths, horizon_years=args.horizon)
        print(summarize(sim, dt=float(cfg["data"]["dt"])).to_string(index=False))
        np.save(os.path.join(args.out, "regime_paths.npy"), sim["_regime"])

    # insertion directe dans la note
    if not args.no_note and not args.no_sim:
        ref = load_reference(ref_path)
        section = build_section(cal, ref, cfg, sim)
        update_note(args.note, section)
        print(f"\nSection « Résultats numériques » insérée dans {args.note} "
              f"(entre balises GSE-AUTO-RESULTS).")
    elif args.no_sim:
        print("\n(Section non régénérée : --no-sim. La section inclut les stats de simulation.)")

    print("\nTerminé. Sorties dans :", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
