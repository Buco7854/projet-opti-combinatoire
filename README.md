# CVRP : Optimisation par Colonies de Fourmis améliorée par Q-Learning

**Projet d'Optimisation Combinatoire** — Apport du Machine Learning pour l'amélioration des métaheuristiques

**Groupe** : Grégory MENEUS, Arnaud GRIMBERT, Odin LANDU

---

## Description du projet

Ce projet traite le **CVRP** (Capacitated Vehicle Routing Problem) en combinant :

- **Métaheuristique** : Optimisation par Colonies de Fourmis (ACO)
- **Machine Learning** : Apprentissage par Renforcement via Q-Learning tabulaire

Le Q-Learning ajuste dynamiquement les paramètres de l'ACO (alpha, beta, rho) au cours de la résolution, en observant l'état de la recherche (taux d'amélioration, diversité des solutions, phase de la recherche).

### Le problème CVRP

Le CVRP consiste à organiser les itinéraires optimaux d'une flotte de véhicules partant d'un dépôt central pour livrer des clients ayant des demandes spécifiques, sans dépasser la capacité maximale par véhicule. L'objectif est de minimiser la distance totale parcourue.

### Approche hybride

1. **ACO classique** : les fourmis construisent des solutions guidées par les phéromones et l'heuristique de distance
2. **Q-Learning** : observe l'état de la recherche et ajuste les paramètres ACO
   - **États** : taux d'amélioration × diversité × phase de recherche (27 états)
   - **Actions** : augmenter/diminuer alpha, beta, rho, ou ne rien changer (7 actions)
   - **Récompense** : amélioration relative du meilleur coût

---

## Structure du dépôt

```
.
├── src/
│   ├── cvrp.py              # Modélisation du problème CVRP
│   ├── aco.py               # ACO classique
│   ├── qlearning.py         # Agent Q-Learning
│   ├── hybrid_aco_ql.py     # Solveur hybride ACO + Q-Learning
│   └── experiments.py       # Lancement des expériences
├── templates/
│   └── index.html           # Interface web
├── results/                  # Résultats expérimentaux
├── app.py                    # Serveur Flask (démo web)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Instructions d'exécution

### Prérequis

- Python 3.10+
- pip

### Installation locale

```bash
pip install -r requirements.txt
```

### Lancer les expériences

```bash
python -m src.experiments
```

Les résultats sont sauvegardés dans `results/results_summary.json`.

### Lancer l'interface web

```bash
python app.py
```

Puis ouvrir http://localhost:5000 dans un navigateur.

### Avec Docker

```bash
docker-compose up --build
```

L'application est accessible sur http://localhost:5000.

---

## Résultats expérimentaux

Les expériences comparent l'ACO classique et l'ACO + Q-Learning sur 7 instances de tailles croissantes (10 à 50 clients), avec 5 exécutions par instance.

Les résultats détaillés sont dans `results/results_summary.json`.

---

## Technologies utilisées

- **Python** : langage principal
- **NumPy** : calculs numériques
- **Flask** : interface web de démonstration
- **Chart.js** : visualisation des résultats dans le navigateur
- **Docker** : déploiement simplifié
