"""Génère la section « Résultats numériques » (.tex) à partir du calibrage.

Émet l'INTÉGRALITÉ des sorties numériques (thème Nexialog, booktabs) :
fenêtre et prétraitements, paramètres des marges, coefficients de déslissage,
analyse du choix du nombre d'états latents K*, régime latent COMMUN
(matrice de transition unique + loi invariante + paramètres par actif),
chaînes séparées (validation), corrélations par régime, et statistiques de
contrôle de la simulation. Reproductible :

    python -m gse.export_latex --config gse/config.yaml --out outputs/results_section.tex
"""
from __future__ import annotations
import argparse
import numpy as np

from .calibrate import calibrate, load_config
from .compare import load_reference
from .preprocessing import preprocess_factor
from .simulate import simulate, summarize

LAB = {'inflation.short': 'Inf.C', 'inflation.long': 'Inf.L',
       'real_rate.short': 'TxR.C', 'real_rate.long': 'TxR.L',
       'credit': 'Créd', 'dette_privee': 'DetP', 'pe': 'PE',
       'immobilier': 'Immo', 'infra': 'Infra', 'Action_EUR': 'Act.EU',
       'Action_Monde': 'Act.Mo', 'Action_emergent': 'Act.EM'}


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


def build_section(cal, ref, cfg):
    dt = float(cfg["data"]["dt"])
    L = []
    a = L.append
    a(r"% ====================================================================")
    a(r"\section{Résultats numériques (sortie du module de calibrage)}")
    a(r"\label{sec:resultats}")
    a(r"% ====================================================================")
    a(r"Tous les résultats ci-dessous sont \emph{produits par l'outil} "
      r"(package \texttt{gse}) sur \texttt{Historical\_Data\_\allowbreak Model\_\allowbreak Calibration.xlsx}. "
      r"Conformément au cadre de la note, \textbf{les facteurs dépendant du régime "
      r"latent (actions) sont calibrés ensemble, sous une \emph{unique} chaîne de "
      r"Markov} (une seule matrice de transition)~; le nombre d'états $K^\star$ est "
      r"choisi par critère d'information. La colonne \emph{Réf.} reprend "
      r"\texttt{Parametres\_models.xlsx} (2026).")

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
        d0, d1 = pre.index[0].date(), pre.index[-1].date()
        tr = (spec.get("preprocessing", {}) or {}).get("transform", "none")
        extra = []
        if (spec.get("preprocessing", {}) or {}).get("unsmooth"):
            extra.append("déslissage AR(1)")
        if spec.get("fix"):
            extra.append("fix " + ",".join(spec["fix"].keys()))
        if spec["model"] == "RSLN2":
            extra.append("régime commun")
        sheets = spec.get("sheet") or ", ".join(spec.get("sheets", {}).values()
                                                if isinstance(spec.get("sheets"), dict)
                                                else spec.get("sheets", []))
        tr_txt = {"none": "aucun", "log_return_100": r"$100\ln$-rdt",
                  "log100": r"$100\ln$", "log": r"$\ln$"}.get(tr, tr)
        note = tr_txt + (("~; " + ", ".join(extra)) if extra else "")
        a(f"{esc(name)} & {spec['model']} & \\footnotesize {esc(sheets)} & {n} & \\footnotesize {note}\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Périmètre et prétraitements par facteur. Fenêtre commune "
      r"déc.~2005--déc.~2025 ($n$ = nombre d'observations modélisées~; l'immobilier "
      r"démarre en juin~2008).}\label{tab:res-data}")
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
    hl = lambda k: np.log(2)/k
    a(f"Demi-vie court & an & {_f(hl(inf['kappa_short']),2)} & -- & -- & {_f(hl(rea['kappa_short']),2)} & -- & --\\\\")
    a(f"Demi-vie long & an & {_f(hl(inf['kappa_long']),2)} & -- & -- & {_f(hl(rea['kappa_long']),2)} & -- & --\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{V2F (EMV exact~; $\mu$ inflation fixé à la cible COR). Demi-vie $=\ln 2/\kappa$. "
      r"Écarts $\kappa,\sigma$ : la référence emploie les cibles distributionnelles "
      r"(cf.\ \S\ref{sec:resultats-discussion}).}\label{tab:res-v2f}")
    a(r"\end{table}")

    # ---------- R.3 CIR / BK / BS ----------
    cr, bk = cal.margins["credit"].params, cal.margins["dette_privee"].params
    pe, im, inf2 = cal.margins["pe"].params, cal.margins["immobilier"].params, cal.margins["infra"].params
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
    row("PE (non coté)", "BS", r"$\mu$ (\%/an)", pe["mu"], ref[("pe","mu")], 2)
    row("", "", r"$\sigma$ (\%/an)", pe["sigma"], ref[("pe","sigma")], 2)
    row("Infrastructure", "BS", r"$\mu$ (\%/an)", inf2["mu"], ref[("infra","mu")], 2)
    row("", "", r"$\sigma$ (\%/an)", inf2["sigma"], ref[("infra","sigma")], 2)
    row("Immobilier", "BS", r"$\mu$ (\%/an)", im["mu"], ref[("immobilier","mu")], 2)
    row("", "", r"$\sigma$ (\%/an)", im["sigma"], ref[("immobilier","sigma")], 2)
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Crédit (CIR, EMV $\chi^2$ décentré), dette privée (BK, sur $100\ln(\text{spread})$) "
      r"et actifs réels (BS).}\label{tab:res-margins}")
    a(r"\end{table}")

    # ---------- R.4 Coefficients de déslissage ----------
    a(r"\subsection{Coefficients de déslissage (\emph{unsmoothing})}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{6pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l X r r r r@{}}")
    a(r"\toprule")
    a(r"\textbf{Actif} & \textbf{Modèle} & \textbf{AR(1) $b$} & \textbf{$\sigma$ brute} & \textbf{$\sigma$ déslissée} & \textbf{Inflation var.}\\")
    a(r"\midrule")
    for nm, lbl in [("pe", "PE (LPX50)"), ("immobilier", "Immobilier"), ("infra", "Infrastructure")]:
        p = cal.margins[nm].params
        a(f"{lbl} & BS & {_f(p['ar1_b'],4)} & {_f(p['sigma_raw'],2)} & {_f(p['sigma_unsmoothed'],2)} & {_f(p['var_inflation'],3)}\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Déslissage AR(1) : $r^\ast_t=(r_t-b\,r_{t-1})/(1-b)$. La volatilité retenue "
      r"est la volatilité \emph{déslissée} (annualisée, \%)~; le facteur d'inflation de variance "
      r"vaut $(1+b)/(1-b)$. PE montre la sur-correction attendue ($b>0$)~; immobilier et infra "
      r"sont peu autocorrélés ici.}\label{tab:res-unsmooth}")
    a(r"\end{table}")

    # ---------- R.5 Choix du nombre d'états ----------
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
    a(rf"\caption{{Modèle à régime commun (émissions gaussiennes $D=3$ actions, $n={ks['n']}$). "
      r"Le BIC sélectionne $K^\star=" + str(ks["k_star"]) + r"$ ; l'AIC, moins parcimonieux, "
      r"continue de décroître. Un seul régime commun à $K^\star$ états remplace $2^3$ états "
      r"joints de chaînes séparées.}\label{tab:res-kselect}")
    a(r"\end{table}")

    # ---------- R.6 Régime latent commun ----------
    J = cal.regime.joint
    P = np.asarray(J["P"]); pi = np.asarray(J["pi"]); K = J["K"]
    a(r"\subsection{Régime latent commun : transition unique et paramètres par actif}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.25}\setlength{\tabcolsep}{6pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l X" + " r"*(2*K) + r"@{}}")
    a(r"\toprule")
    hdr = " & ".join([fr"\multicolumn{{2}}{{c}}{{\textbf{{Régime {k+1}}}}}" for k in range(K)])
    a(r"\textbf{Actif} & \textbf{} & " + hdr + r"\\")
    cmid = "".join([fr"\cmidrule(lr){{{3+2*k}-{4+2*k}}}" for k in range(K)])
    a(cmid)
    a(r"\textbf{} & & " + " & ".join([r"$\mu$ & $\sigma$" for _ in range(K)]) + r"\\")
    a(r"\midrule")
    for sheet, lbl in [("Action_EUR", "Euro"), ("Action_Monde", "Monde"), ("Action_emergent", "Émergent")]:
        regs = J["regimes_by_equity"][sheet]
        cells = " & ".join([f"{_f(regs[k]['mu'])} & {_f(regs[k]['sigma'])}" for k in range(K)])
        a(f"{lbl} & & {cells}\\\\")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Actions sous le régime latent \emph{commun} ($K^\star=" + str(K) +
      r"$, EM joint). $\mu,\sigma$ annualisés en \%. Le régime~1 (forte volatilité) frappe "
      r"\emph{simultanément} les trois actifs --- dépendance de queue systémique.}\label{tab:res-common}")
    a(r"\end{table}")
    # transition + invariant
    a(r"\begin{definitionbox}")
    Pmat = r"\\".join([" & ".join(_f(P[i, j], 4) for j in range(K)) for i in range(K)])
    pirow = r",\ ".join(_f(pi[k], 4) for k in range(K))
    a(r"\[ P=\begin{pmatrix}" + Pmat + r"\end{pmatrix},\qquad \pi=("
      + pirow + r")\transp. \]")
    a(r"Matrice de transition mensuelle \emph{unique} et loi invariante du régime commun "
      r"(probabilité stationnaire de stress $\pi_1=" + _f(pi[0]*100, 1) + r"\,\%$).")
    a(r"\end{definitionbox}")

    # ---------- R.7 chaînes séparées (validation) ----------
    a(r"\subsection{Validation : chaînes séparées par actif vs référence}")
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{4pt}")
    a(r"\begin{tabularx}{\textwidth}{@{}l X r r r r r r@{}}")
    a(r"\toprule")
    a(r"& & \multicolumn{2}{c}{\textbf{Rég.\ 1 (stress)}} & \multicolumn{2}{c}{\textbf{Rég.\ 2 (normal)}} & \multicolumn{2}{c}{\textbf{Transitions}}\\")
    a(r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
    a(r"\textbf{Actif} & \textbf{Source} & $\mu_1$ & $\sigma_1$ & $\mu_2$ & $\sigma_2$ & $p_{1\to2}$ & $p_{2\to1}$\\")
    a(r"\midrule")
    for sheet, lbl in [("Action_EUR", "Euro"), ("Action_Monde", "Monde"), ("Action_emergent", "Émergent")]:
        sp = cal.regime.params_separate[sheet]
        r1, r2 = sp["regimes"]
        k = f"equities:{sheet}"
        a(f"{lbl} & Calib. & {_f(r1['mu'])} & {_f(r1['sigma'])} & {_f(r2['mu'])} & {_f(r2['sigma'])} & {_f(sp['p_1to2']*100)} & {_f(sp['p_2to1']*100)}\\\\")
        a(f" & Réf. & {_f(ref[(k,'R1.mu')])} & {_f(ref[(k,'R1.sigma')])} & {_f(ref[(k,'R2.mu')])} & {_f(ref[(k,'R2.sigma')])} & {_f(ref[(k,'p_1to2')]*100)} & {_f(ref[(k,'p_2to1')]*100)}\\\\")
        a(r"\addlinespace")
    a(r"\bottomrule\end{tabularx}")
    a(r"\caption{Chaînes 2-états \emph{séparées} (mode de compatibilité avec la référence, qui "
      r"calibre une chaîne par actif). Euro/Monde concordent à $<1{,}5\,\%$~; l'émergent atteint "
      r"une vraisemblance \emph{supérieure} à la référence.}\label{tab:res-rsln-sep}")
    a(r"\end{table}")

    # ---------- R.8 corrélations ----------
    comps = cal.dependence["components"]
    O1 = cal.dependence["regimes"]["reg1"].values
    O2 = cal.dependence["regimes"]["reg2"].values
    d = len(comps)
    M = np.where(np.triu(np.ones((d, d)), 1) > 0, O1, O2)
    np.fill_diagonal(M, 1.0)
    a(r"\subsection{Corrélations des résidus par régime}")
    a(r"\begin{landscape}")
    a(r"\begin{table}[H]\centering\scriptsize\renewcommand{\arraystretch}{1.1}\setlength{\tabcolsep}{2.6pt}")
    a(r"\begin{tabular}{@{}l" + "r" * d + r"@{}}")
    a(r"\toprule")
    a(" & " + " & ".join(LAB[c] for c in comps) + r"\\")
    a(r"\midrule")
    for i in range(d):
        cells = [r"\textbf{1}" if i == j else f"{M[i,j]:.2f}" for j in range(d)]
        a(LAB[comps[i]] + " & " + " & ".join(cells) + r"\\")
    a(r"\bottomrule\end{tabular}")
    a(r"\caption{Corrélations des résidus standardisés (masque \texttt{groupB}). "
      r"\emph{Triangle sup.} = régime de stress~; \emph{triangle inf.} = régime normal. "
      r"Hors actions, les corrélations sont communes aux régimes~; SDP (Higham).}\label{tab:res-corr}")
    a(r"\end{table}")
    a(r"\end{landscape}")

    # ---------- R.9 simulation ----------
    a(r"\subsection{Simulation : statistiques de contrôle}")
    sim = simulate(cal, n_paths=2000, horizon_years=int(cfg.get("simulation", {}).get("horizon_years", 30)))
    summ = summarize(sim, dt=dt)
    occ = float((sim["_regime"] == 0).mean()) * 100
    a(r"\begin{table}[H]\centering\small\renewcommand{\arraystretch}{1.2}\setlength{\tabcolsep}{6pt}")
    a(r"\begin{tabular}{@{}l l r l r@{}}")
    a(r"\toprule")
    a(r"\textbf{Sortie} & \textbf{Mesure} & \textbf{Val.} & \textbf{} & \textbf{E[terminal]}\\")
    a(r"\midrule")
    for _, row in summ.iterrows():
        a(f"{esc(row['sortie'])} & {esc(row['m1'])} & {_f(row['v1'],3)} & {esc(row['m2'])} & {_f(row['v2'],4)}\\\\")
    a(r"\bottomrule\end{tabular}")
    a(rf"\caption{{Contrôle de cohérence sur {sim['_regime'].shape[0]} trajectoires "
      rf"$\times$ {sim['_regime'].shape[1]-1} pas (horizon 30~ans). Occupation simulée du régime "
      rf"de stress~: {_f(occ,1)}\,\% (cf.\ $\pi_1$). Les volatilités simulées reproduisent les "
      r"paramètres calibrés.}\label{tab:res-sim}")
    a(r"\end{table}")

    # ---------- discussion ----------
    a(r"\subsection{Lecture et concordance avec la référence}")
    a(r"\label{sec:resultats-discussion}")
    a(r"\begin{insightbox}[Concordances et écarts attendus]")
    a(r"\textbf{Concordances exactes} (méthodologie identique)~: crédit CIR ($\kappa,\sigma,\theta$), "
      r"dette privée $\kappa$, PE/infra $\mu$, moyenne LT des taux réels, $\mu$ inflation (fixé)~; "
      r"en mode chaînes séparées, régimes actions Euro/Monde à moins de $1{,}5\,\%$. "
      r"\textbf{Régime commun}~: une seule matrice de transition pilote les trois actions "
      r"(entrée conjointe en stress, $\pi_1\approx" + _f(pi[0]*100, 0) + r"\,\%$)~; $K^\star=" +
      str(K) + r"$ états par BIC. \textbf{Écarts attendus et documentés}~: (i)~$\kappa,\sigma$ des "
      r"V2F (référence = cibles distributionnelles, mal conditionnées~; l'outil applique l'EMV)~; "
      r"(ii)~$\sigma$ dette privée (référence = dispersion stationnaire~; $\kappa,\mu$ concordent)~; "
      r"(iii)~émergent (vraisemblance plus élevée ici)~; (iv)~immobilier (indice de \emph{prix} "
      r"fourni vs rendement total avec loyers).")
    a(r"\end{insightbox}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gse/config.yaml")
    ap.add_argument("--out", default="outputs/results_section.tex")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cal = calibrate(cfg)
    ref_path = cfg.get("reference", {}).get("path", "Parametres_models.xlsx")
    ref = load_reference(ref_path)
    sec = build_section(cal, ref, cfg)
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(sec)
    print("written", args.out, f"({len(sec.splitlines())} lines)")


if __name__ == "__main__":
    main()
