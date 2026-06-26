# Audit du code GSE (`gse/`) — conformité théorique et points d'attention

Audit complet du package `gse` et du pilote `run_calibration.py`, confronté au
cadre théorique de la note *« Calibrage et simulation du GSE — représentation
autorégressive à régime latent commun »*. Chaque constat est noté en sévérité
**[Critique] / [Majeur] / [Modéré] / [Mineur] / [Cosmétique]**, localisé, étayé
(lecture de code + vérifications empiriques), et assorti d'une recommandation.

**Verdict global.** L'implémentation est **fidèle au cadre théorique** : forme
AR universelle, résidus PIT, MS-VAR à régime *commun* unique, cascade IFM en
trois étapes, sélection de $K^\star$, simulation cohérente. Les constats portent
surtout sur (i) la **robustesse en échantillon fini** de la matrice de
dépendance par régime, (ii) l'**alignement des dates**, (iii) l'**adéquation**
(normalité des résidus), et quelques **nuances de cohérence** ; aucun défaut ne
remet en cause l'architecture.

---

## 1. Conformité au cadre théorique (note ↔ code)

| Concept de la note | Implémentation | Conforme |
|---|---|---|
| Forme AR universelle $Y_{t+\Delta}=\Phi Y_t+c+D\varepsilon$ | `margins.py`, `regime.py` | ✔ |
| Discrétisation exacte V2F (EMV = MCO) | `fit_v2f` (long OU + court sur $(1,q^s,q^l)$) | ✔ |
| $\sigma_1$ net de la variance transmise | `fit_v2f` (terme $\mathcal B$) | ✔ |
| CIR : init Euler + EMV $\chi^2$ décentré ; PIT | `fit_cir` | ✔ |
| BK : OU exact sur log-spread | `fit_bk` | ✔ |
| BS : moyenne brute + vol déslissée | `fit_bs` | ✔ |
| Résidus PIT, standardisation composante par composante, $\rho_c$ dans $\Omega$ | margins + `dependence.py` | ✔ (vérifié : $\rho_c$ résidus = modèle) |
| **Régime latent COMMUN unique** (une seule $P$) | `fit_rsln2` (joint), `simulate` | ✔ |
| Choix de $K^\star$ par BIC | `select_k` | ✔ |
| $\Omega(a)$ par régime sur résidus, masque de sensibilité, SDP Higham | `compute_dependence` | ✔ (voir F1, F3) |
| Simulation : régime commun, $\varepsilon=L(a)z$, cartes par facteur | `simulate` | ✔ (voir F4, F6) |

---

## 2. Constats

### [Majeur] F1 — Matrice de corrélation du régime de stress quasi singulière
`dependence.py: compute_dependence`. La $\Omega(\text{reg}_1)$ (stress) a une
**valeur propre minimale $\approx 10^{-10}$** (rang déficient) ; seule la
projection de Higham (plancher $10^{-10}$) la rend inversible, rendant le
Cholesky et la simulation **numériquement fragiles** et les corrélations de
stress peu fiables. Cause : la matrice $12\times12$ (66 corrélations) est estimée
sur un échantillon effectif minuscule — le régime de stress occupe $\approx 29\%$
de $\sim148$ points communs, soit $\sim43$ points pondérés.
*Évidence* : `min_eig(reg1)=0.0000` vs `reg2=0.0354`.
**Reco** : régularisation par **rétrécissement** (Ledoit–Wolf) de $\Omega(a)$
vers la cible poolée ou l'identité, en particulier pour les régimes peu peuplés ;
ou plancher de valeur propre explicite (> $10^{-10}$) ; documenter l'incertitude.

### [Majeur] F2 — Désalignement des dates → dépendance sous-échantillonnée
`dependence.py` intersecte des **timestamps exacts**. Or les feuilles ont des
jours-de-mois hétérogènes (26–31). L'intersection brute des séries du Groupe A +
actions tombe à **148 points**, alors que l'alignement **par mois** en donne
**211** (perte $\approx 30\%$). $\Omega$ est donc estimée sur bien moins de
données que disponible (et cela aggrave F1).
*Évidence* : intersection timestamps = 148 ; par `to_period("M")` = 211.
**Reco** : aligner toutes les séries sur une période mensuelle (`to_period("M")`
ou rééchantillonnage fin de mois) **avant** intersection, dans `preprocessing`
et/ou `dependence`.

### [Majeur] F3 — Résidus PIT non parfaitement gaussiens (adéquation)
Sous bonne spécification, la note pose $\widehat\varepsilon\sim\Ncal(0,1)$ et une
copule gaussienne. Les tests KS rejettent la normalité pour plusieurs facteurs :
**crédit $p=0{,}001$**, immobilier $0{,}005$, PE $0{,}007$, inflation court
$0{,}026$, dette privée $0{,}031$ (les autres OK). Le crédit (CIR) montre aussi
$\text{std}=0{,}981$. Il s'agit d'une limite d'**adéquation** (queues épaisses),
pas d'un défaut de code.
**Reco** : volet validation (Phase 2) — tests de spécification, envisager des
marges/copules à queues (Student) ; à défaut, documenter l'hypothèse.

### [Modéré] F4 — Incohérence estimation/simulation pour le CIR
Estimation : PIT **exacte** ($\chi^2$ décentré). Simulation : par défaut
`cir_scheme: alfonsi` (`config.yaml`), dont la **marge n'est qu'approchée** →
la loi simulée du crédit diffère légèrement de la loi estimée (la copule
gaussienne par régime n'est exactement respectée qu'avec l'inverse-PIT).
`simulate.py` implémente déjà `inverse_pit` (exact).
**Reco** : passer `cir_scheme: inverse_pit` pour la cohérence stricte, ou
documenter l'approximation Alfonsi (rapide).

### [Modéré] F5 — V2F : facteurs identifiés aux taux observés 2A/10A
`preprocessing.preprocess_factor` (V2F) prend `Inflation_2Y` comme $q^s$ et
`Inflation_10Y` comme $q^l$ : les facteurs sont **identifiés aux taux observés**,
pas à des facteurs latents affines (toutes maturités chargeant les deux facteurs).
C'est cohérent avec l'EMV de la note, mais constitue une hypothèse de modélisation
distincte du modèle affine de la référence (cibles distributionnelles), ce qui
explique les écarts $\kappa,\sigma$.
**Reco** : l'expliciter (déjà partiellement dans la note) ; proposer en option un
filtre de Kalman affine si l'on veut des facteurs latents.

### [Mineur] F6 — Décalage d'indice du régime en simulation
`simulate.py` : au pas $t$, le choc et le rendement utilisent le régime
**courant $E_{t-1}$** (avant transition), tandis que l'EM indexe l'émission
$x_t$ sous $E_t$. Sans effet sur la loi (chaîne stationnaire, init $\pi$ —
vérifié : occupation simulée du stress $29{,}0\%$ vs $\pi_1=29{,}1\%$), mais
convention différente de l'estimation.
**Reco** : documenter, ou tirer la transition **avant** le rendement.

### [Vérifié — OK] F7 — Estimateur de variance $1/n$ cohérent note ↔ code
Contrôlé : `fit_ou_scalar`/`fit_bs` utilisent $1/n$ (`ddof=0`, EMV) et la note
écrit aussi $\hat v=\frac1n\sum\widehat\xi^2$ (boîte (A), et BS). **Cohérent,
aucune action.**

### [Mineur] F8 — Reconstruction du spread BK codée en dur (`exp(y/100)`)
`simulate.py` reconstruit le spread par `exp(y/100)`, supposant l'échelle
$100\ln$. Correct pour la dette privée (seul BK, `transform: none` sur une série
déjà $100\ln$), mais non générique (un BK avec `transform: log` donnerait
`exp(y)`). `sim["transform"]` est stocké mais inutilisé.
**Reco** : choisir `exp(y)` vs `exp(y/100)` selon `sim["transform"]`.

### [Mineur] F9 — BS : sources de moyenne et de volatilité distinctes
`fit_bs` : moyenne sur rendements **bruts**, volatilité sur rendements
**déslissés** ; le résidu (pour $\Omega$) standardise la série de volatilité. Le
déslissage étant approximativement préservateur de moyenne, l'incohérence est
négligeable, mais c'est un mélange de sources.
**Reco** : documenter ; vérifier l'écart de moyenne brute/déslissée.

### [Mineur] F10 — Garde-fou $\beta$ du OU
`fit_ou_scalar` borne $\hat\beta\in(10^{-8},1-10^{-10})$. Un $\hat\beta<0$
(anti-persistance, bruit) serait écrêté à $10^{-8}$ → $\hat\kappa$ aberrant. Non
observé sur ces données (taux persistants, $\beta\approx0{,}97$), mais silencieux.
**Reco** : émettre un avertissement si $\hat\beta\le0$ ou $\ge1$.

### [Mineur] F11 — Re-centrage global (et non par régime) des résidus Groupe A
`compute_dependence` centre $EA$ par la moyenne **globale** sur l'échantillon
commun ; pour $\Omega^{AB}(a)$ pondéré par régime, le centrage devrait être
conditionnel. Biais négligeable (résidus $\approx$ centrés), mentionné pour
exhaustivité.

### [Mineur] F12 — Régime porté uniquement par les actions
Seules les 3 actions portent le régime commun ; PE/immo/infra restent Groupe A
(insensibles), conforme à la référence mais la note les dit « promouvables ».
**Reco** : documenter le choix ; le rendre configurable (les inclure au Groupe B).

### [Cosmétique] F13 — Sorties JSON incomplètes ; warning pandas
`calibrate.params_table` exporte `equities_separate` (désormais vide) et n'inclut
pas `regimes_by_equity` (paramètres par actif du régime commun). Un
`FutureWarning` pandas (concat sans `sort`) apparaît dans `dependence`/checks.
**Reco** : dumper `regimes_by_equity` + `P`, $\pi$ ; préciser `sort=False`.

### [Mineur] F14 — Immobilier : donnée prix vs rendement total
$\mu,\sigma$ immobilier ne reproduisent pas la référence (indice de **prix** vs
rendement total avec loyers réinvestis ; `income_yield=0`). **Donnée**, pas code ;
déjà documenté.

---

## 3. Vérifications empiriques (preuves)

| Contrôle | Résultat | Statut |
|---|---|---|
| Résidus PIT : moyenne / écart-type | $\approx 0$ / $\approx 1$ partout | ✔ |
| Résidus PIT : normalité (KS) | rejet pour crédit/PE/immo/inflation court | ⚠ F3 |
| $\Omega(a)$ symétrique, diag $=1$ | oui | ✔ |
| $\Omega$ SDP (val.\ propre min) | reg2 $=0{,}035$ ; **reg1 $\approx 0$** | ⚠ F1 |
| V2F : corr résidus $=\rho_c$ modèle | infl $0{,}805/0{,}805$ ; réel $0{,}544/0{,}544$ | ✔ |
| Loi invariante $\pi$ = stationnaire de $P$ | $(0{,}291,0{,}709)$ identiques | ✔ |
| Régime de stress systémique | moyennes des 3 actions $<0$ en reg1 | ✔ |
| Simulation : vol annualisée ≈ calibrée | PE 29,6≈29,6 ; infra 13,4≈13,4 | ✔ |
| Simulation : positivité CIR / spread | min $>0$ (Alfonsi/exp) | ✔ |
| Simulation : retour à la moyenne crédit | $E[\text{term}]=1{,}44\approx\theta=1{,}42$ | ✔ |
| Occupation simulée du régime de stress | $29{,}0\%\approx\pi_1=29{,}1\%$ | ✔ |
| Échantillon de la dépendance | 148 (timestamps) vs 211 (mensuel) | ⚠ F2 |

---

## 4. Robustesse, numérique et reproductibilité

- **EM** : multi-démarrage (20), planchers de variance, tri des états par
  volatilité (anti *label-switching*), filtre Hamilton + lisseur Kim, scaling
  log-somme. Robuste. (Vérifié : l'émergent atteint une vraisemblance > référence.)
- **Higham** : projections alternées (Dykstra), plancher de valeurs propres.
  Correct, mais cf.\ F1 (régime stress).
- **Log matriciel V2F** : `fit_v2f` procède équation par équation (pas de `logm`),
  donc pas de risque de log complexe ; cohérent avec la note (simplification
  consistante).
- **Reproductibilité** : graine fixée pour la simulation ; **mais** l'EM dépend de
  graines internes fixes (`seed=...`) — reproductible. Les sorties Monte-Carlo
  varient avec `n_paths` (estimation MC), normal.
- **Validation des entrées** : `preprocessing` lève des erreurs explicites
  (séries vides, non positives pour log/CIR, fenêtre trop courte). Bon.
- **Cohérence options** : $\mathcal F$ (paramètres fixés) et masque $S$ s'appliquent
  identiquement à estimation, résidus et simulation. Vérifié pour $\mu_l=1{,}75$.

---

## 5. Recommandations prioritaires

1. **(F2)** Aligner les séries par **mois** avant l'intersection de la dépendance
   → +30 % de points (148→211). Effet immédiat sur la qualité de $\Omega$.
2. **(F1)** **Rétrécir** $\Omega(a)$ par régime (Ledoit–Wolf vers la cible poolée)
   pour lever la quasi-singularité du régime de stress ; plancher de valeur propre
   explicite.
3. **(F4)** Basculer la simulation du CIR sur `inverse_pit` pour la cohérence
   estimation/simulation exacte (ou documenter Alfonsi).
4. **(F3)** Volet d'adéquation (Phase 2) : tests de normalité/queues sur les
   résidus PIT ; envisager une copule de Student si le risque de queue est clé.
5. **(F8,F13)** Corrections de forme : généraliser `exp(y/100)` selon le
   `transform`, enrichir le dump JSON (ajouter `regimes_by_equity`), préciser
   `sort=False` (warning pandas).

Aucune de ces actions ne modifie l'architecture ; elles renforcent la robustesse
et la cohérence fine. Le cœur (calibrage par vraisemblance, régime commun unique,
dépendance par régime, simulation) est conforme et vérifié.
