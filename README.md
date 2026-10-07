# Forex Correlation Alert

Surveillance Forex **alert-only**, sans aucune exécution automatique.

## Version actuelle

Le moteur combine une ancre **H4** avec des confirmations **H1/M15**, puis filtre les relations instables avant de générer une alerte.

### Détection

- données intraday M15 via Yahoo Finance Chart ;
- construction M15 → H1 → H4 avec bougies alignées sur leur clôture ;
- corrélation de Pearson sur rendements ;
- seuil configurable **|r| >= 0.65** ;
- H4 obligatoire comme timeframe primaire ;
- contrôle de stabilité sur 4 segments ;
- contrôle de dérive entre corrélation courte et longue ;
- direction EMA 20/50 ;
- confluence H4 → H1 → M15 ;
- score sur 100 ;
- seuil de confluence configurable (`H4_ONLY` par défaut) ;
- état persistant `NEW / MAINTAINED / CLEARED` ;
- rapports JSON et Markdown ;
- workflow GitHub Actions toutes les 30 minutes ;
- aucune fonction d'ordre et aucune clé de trading.

## Score

- corrélation : 40 points ;
- H4 : 25 points ;
- H1 : 20 points ;
- M15 : 15 points.

Score minimal par défaut : **65/100**.

## Audit historique

Le workflow `Forex Correlation Backtest` réalise un audit historique indépendant de toute exécution.

Il compare :

1. une baseline brute basée uniquement sur `|r| >= 0.65` ;
2. le modèle filtré par stabilité, dérive, confluence et score.

Les résultats sont mesurés à **4 h, 8 h, 16 h et 24 h**.

L'audit ne calcule pas de P&L : il mesure la persistance de la relation entre les deux devises. Les signaux utilisent des bougies H4 clôturées et les résultats commencent sur les bougies suivantes afin d'éviter le look-ahead.

Sorties principales :

- `backtest_raw_observations.csv`
- `backtest_observations.csv`
- `backtest_breakdown.csv`
- `backtest_summary.json`
- `backtest_summary.md`

## Sorties live

Chaque exécution produit :

- `results/latest_alerts.json`
- `results/latest_alerts.md`
- `results/alert_state.json`

Le rapport live affiche notamment :

- corrélation H4/H1/M15 ;
- confluence ;
- stabilité ;
- dérive de régime ;
- score ;
- statut `NEW` ou `MAINTAINED`.

## Structure

```text
.github/workflows/
  forex-alert.yml
  forex-backtest.yml
config/config.yaml
src/forex_alert/
  config.py
  data.py
  correlation.py
  scoring.py
  alerts.py
  state.py
  report.py
  main.py
  backtest.py
tests/test_smoke.py
requirements.txt
README.md
```

## Sécurité

Le projet ne contient aucun endpoint d'exécution d'ordre, aucun secret de trading et aucune logique de positionnement automatique.
