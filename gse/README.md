# GSE IRCANTEC — Prétraitement, calibrage et simulation

Implémentation du cadre de la note de recherche
*« Calibrage et simulation du GSE — représentation autorégressive à régime
latent commun »* : prétraitement piloté par configuration, calibrage par
**maximum de vraisemblance** (séquentiel, IFM), dépendance **pleinement
estimée par régime**, et simulation Monte-Carlo cohérente.

---

## 1. Installation

```bash
pip install -r gse/requirements.txt
```

Les deux classeurs de données sont attendus à la racine du dépôt
(fournis sur la branche `main`) :
`Historical_Data_Model_Calibration.xlsx` et `Parametres_models.xlsx`.

## 2. Exécution

```bash
python run_calibration.py                 # calibrage + comparaison + simulation
python run_calibration.py --no-note       # sans modifier la note LaTeX
python run_calibration.py --paths 5000 --horizon 30
```

Sorties (`outputs/`) : `parametres_calibres.json`,
`comparaison_parametres.csv`, `correlation_reg{1,2}.csv`, `regime_paths.npy`.

**Insertion automatique dans la note.** `run_calibration.py` génère la section
« Résultats numériques » (tableaux de paramètres, choix de $K^\star$,
déslissage, régime commun, corrélations, simulation) et l'**insère
directement** dans `Note_recherche_GSE_calibrage_simulation.tex`, entre les
balises `% >>> GSE-AUTO-RESULTS BEGIN ... >>>` et `% <<< GSE-AUTO-RESULTS END
<<<` (créées automatiquement à la première exécution). Utiliser `--no-note`
pour désactiver.

## 3. Architecture (modulaire)

| Module | Rôle |
|---|---|
| `config.yaml` | **source unique** : par facteur, modèle + prétraitement + paramètres fixés + sensibilité au régime |
| `preprocessing.py` | chargement Excel, fenêtrage, 100·log-rendements, log/log100, déslissage AR(1) |
| `margins.py` | calibrateurs Groupe A : `V2F`, `CIR`, `BK`, `BS` (EMV / formes fermées) |
| `regime.py` | `RSLN2` (Hardy) par EM robuste : chaînes séparées **ou** régime commun |
| `dependence.py` | Ω(a) par régime sur résidus PIT + masque de sensibilité + Higham SDP |
| `calibrate.py` | orchestrateur (cascade séquentielle IFM en 3 étapes) |
| `simulate.py` | simulation MS-VAR (régime commun, chocs `L(a)·z`, cartes par facteur) |
| `compare.py` | comparaison aux paramètres de référence |

**Modularité par classe de modèle.** Plusieurs facteurs de la même classe
se déclarent simplement comme plusieurs entrées de config : ici `inflation`
et `real_rate` sont deux `V2F` traités par le même calibrateur. Ajouter un
3ᵉ V2F = ajouter une entrée `factors:`.

## 4. Prétraitements (par facteur, dans `config.yaml`)

- **V2F** (inflation, taux réels) : séries de taux en niveau ; moyenne LT
  *fixable* (`fix: {mu: 1.75}` pour la cible COR de l'inflation).
- **CIR** (crédit) : spread positif, utilisé tel quel.
- **BK** (dette privée) : OU sur le **log-spread**. La série fournie est
  *déjà* `100·log(spread)` → `transform: none` ; pour un spread brut,
  utiliser `transform: log100`.
- **BS** (PE, immo, infra) : `transform: log_return_100` (100·log-rendement
  d'indice) ; **déslissage AR(1)** ; moyenne sur rendements bruts, **vol sur
  rendements déslissés** ; `income_yield` optionnel (réinvestissement des
  loyers, immobilier). **Promotion au Groupe B** possible via
  `regime_sensitive_params: true` : μ/σ et corrélations deviennent propres au
  régime latent commun (le facteur rejoint l'émission jointe de l'EM).
- **RSLN2** (actions) : 100·log-rendements ; `common_regime` (false = une
  chaîne par actif comme la référence ; true = régime latent commun de la
  note) ; `em_restarts` (multi-démarrage).

## 5. Deux options de calibrage (cf. note, §10)

1. **Fixer des paramètres** (`fix:` par facteur) — vraisemblance profilée :
   les paramètres listés sont imposés, les autres estimés. Démontré ici par
   l'épinglage de `inflation.mu = 1.75` (cible COR), reproduit à 0 %.
2. **Corrélations insensibles au régime** (`correlations.regime_sensitivity`) :
   `full` (tout par régime), `none` (une seule Ω), `groupB` (seules les
   corrélations impliquant les actions varient selon le régime — défaut,
   cohérent avec les entrées « …fort vol » de la référence).
3. **Robustesse au changement de données** (cadre général) : dates
   canonicalisées en fin de mois (alignement inter-facteurs déterministe) ;
   `correlations.shrinkage` (`auto`/`none`/valeur) contre la quasi-singularité
   des régimes peu peuplés ; `simulation.cir_scheme` (`inverse_pit` par défaut,
   exact ; `alfonsi` avec bascule auto si sa condition est violée) ;
   `simulation.copula` (`gaussian`/`student` pour la dépendance de queue) ;
   diagnostics PIT auto (`outputs/diagnostics_pit.csv`).

## 6. Méthodologie de calibrage

Cascade séquentielle (IFM) :

1. **Groupe A** — V2F = **EMV joint pur** sur la loi gaussienne exacte (court &
   long ensemble ; `method: mle` par défaut, options `ols` / `distributional`) ;
   BK = OU exact (EMV = MCO) ; BS = moments ; CIR = init Euler puis EMV χ²
   décentré (option `ols` = Euler seul). Résidus PIT.
2. **Groupe B** — EM / Baum-Welch (filtre de Hamilton + lisseur de Kim),
   multi-démarrage, planchers de variance, tri des états par volatilité.
3. **Dépendance** — Ω(a) = corrélation des résidus standardisés pondérée par
   les probabilités lissées du régime commun, masque de sensibilité,
   rétrécissement (régimes peu peuplés) puis projection SDP (Higham).

## 7. Validation contre `Parametres_models.xlsx`

|écart| médian ≈ **6,0 %** (méthode V2F par défaut = **EMV pur**, qui s'écarte
*volontairement* de la référence distributionnelle) ; correspondances
**exactes** où la méthodologie coïncide :

| Facteur | Résultat |
|---|---|
| Crédit (CIR) | κ, σ, θ **exacts** (0 %) |
| Dette privée (BK) | κ **exact** ; μ à 1,5 % |
| PE / Infra (BS) | μ **exacts**, σ à ≈0,2 %/0,9 % |
| Inflation / Taux réel (V2F) | EMV pur : σ plus faibles que la réf. (cf. comparaison des 3 méthodes) |
| Actions Euro | régime proche de la réf. (μ, σ à ≈1–8 %) |
| Actions Monde / émergent | s'écartent (jusqu'à ≈30 %) : régime **commun** partagé |
| Inflation `mu` (fixé) | **exact** (0 %) |

> **Trois méthodes V2F** (`method: mle | ols | distributional`) sont sorties
> côte à côte avec la référence (`outputs/comparaison_methodes_v2f.csv`). Le
> défaut `mle` (EMV pur, court & long ensemble) restitue des σ **plus faibles** ;
> la méthode `distributional` (cibles, Phase 1) reproduit la référence de près
> (taux réels quasi exact). Le |écart| médian (≈ 6 %) est donc porté par ce
> choix méthodologique (EMV ≠ cibles), par l'immobilier (prix vs rendement
> total) et par le **partage du régime** des actions — tous documentés.

### Écarts attendus et leur cause (documentés)

- **V2F κ_long, σ (inflation, taux réels)** : la référence calibre par
  *cibles distributionnelles* (méthode Phase 1). La note démontre que cet
  objectif est **mal conditionné** (crête σ²/κ) et **biaisé** (κ↓, σ↑) ; cet
  outil utilise l'**EMV** (bien posé) — d'où σ plus faible / κ différent.
  C'est précisément l'amélioration méthodologique recommandée par la note.
  La méthode distributionnelle n'est volontairement **pas** ré-implémentée
  (ill-posée).
- **Dette privée σ** : même nature (la référence vise une dispersion
  stationnaire) ; l'EMV restitue la volatilité d'innovation. κ et μ
  concordent.
- **Actions Monde / émergentes** : sous le **régime latent commun**, les
  caractéristiques par régime (μ, σ) de Monde et émergent s'écartent de la
  référence *par actif* (jusqu'à ≈30 %). C'est l'effet **recherché** du
  partage d'une **unique** chaîne de transition entre les trois actions
  (entrée en stress simultanée), non un défaut de calage : la référence
  calibre une chaîne distincte par actif. Euro, dominant dans la dynamique
  commune, reste proche de la référence. Cf. note, tableau « Régimes :
  moyennes et volatilités vs référence ».
- **Immobilier** : la série fournie est un **indice de prix** (1000→943) ;
  la référence suppose un **rendement total** avec réinvestissement des
  loyers (μ≈5,6 %). Renseigner `income_yield` (et fournir l'indice de
  rendement total) pour reproduire la cible.

## 8. Simulation

`simulate.py` déroule la représentation agrégée : tirage du régime commun,
chocs `ε = L(régime)·z`, propagation par facteur (V2F/BK/BS exacts ; CIR par
Alfonsi E(0) ou inverse-PIT exact). Les volatilités simulées reproduisent les
paramètres calibrés et chaque facteur de retour à la moyenne converge vers sa
cible (contrôle de cohérence intégré, `summarize`).
