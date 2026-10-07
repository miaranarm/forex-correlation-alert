# Forex Correlation Alert

Surveillance Forex **alert-only**, sans aucune exécution automatique.

## Moteur V0.3

- données intraday M15 ;
- construction M15 → H1 → H4 ;
- corrélation de Pearson sur rendements ;
- seuil configurable **|r| >= 0.65** ;
- confirmation multi-timeframe ;
- direction EMA 20/50 ;
- score de confluence sur 100 ;
- état persistant des alertes pour distinguer NEW / MAINTAINED / CLEARED ;
- rapport JSON et Markdown ;
- workflow GitHub Actions toutes les 30 minutes ;
- aucune fonction d'ordre et aucune clé de trading.

## Score

- corrélation : 40 points ;
- H4 : 25 points ;
- H1 : 20 points ;
- M15 : 15 points.

Score minimal par défaut : 65/100.

## Sorties

Chaque exécution produit :

- `results/latest_alerts.json`
- `results/latest_alerts.md`
- `results/alert_state.json`

Le workflow conserve ces fichiers comme artifact.

## Structure

```text
.github/workflows/forex-alert.yml
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
tests/test_smoke.py
requirements.txt
README.md
```

## Données

La source initiale est Yahoo Finance Chart. Une source secondaire pourra être ajoutée ultérieurement.

## Sécurité

Le projet ne contient aucun endpoint d'exécution d'ordre, aucun secret de trading et aucune logique de positionnement automatique.