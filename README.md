# Forex Correlation Alert

Système de surveillance et d'alertes Forex basé sur les corrélations et la confluence multi-timeframe.

## Principe

- Corrélation Pearson sur paires Forex : alerte à partir de |r| >= 0.65.
- Analyse multi-timeframe : H4, H1 et M15.
- Scoring de confluence.
- Alertes uniquement : aucune exécution automatique d'ordres.
- Exécution planifiée via GitHub Actions.
- Les paramètres sont centralisés dans config/config.yaml.

## Structure

```
.
├── .github/workflows/forex-alert.yml
├── config/config.yaml
├── src/forex_alert/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── correlation.py
│   ├── scoring.py
│   ├── alerts.py
│   └── main.py
├── tests/
│   └── test_smoke.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Sécurité

Aucune clé de trading n'est utilisée. Le projet ne contient aucune fonction de passage d'ordre.
