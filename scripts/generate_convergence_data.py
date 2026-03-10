"""
Génère les données de convergence pour le graphe du rapport.
Exécute ACO et ACO+QL sur l'instance large_40 et sauvegarde
l'historique de best_cost à chaque itération dans des fichiers .dat.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cvrp import generate_random_instance
from src.aco import ACOSolver, ACOParams
from src.hybrid_aco_ql import HybridACOQLSolver
from src.qlearning import QLParams


def main():
    # Instance large_40 (même seed que dans experiments.py)
    instance = generate_random_instance(
        num_customers=40, capacity=120, max_coord=100, max_demand=25, seed=47
    )

    aco_params = ACOParams(
        num_ants=20, alpha=1.0, beta=3.0, rho=0.1, Q=100.0, max_iterations=150
    )
    ql_params = QLParams(
        learning_rate=0.2, discount_factor=0.95, epsilon=0.15,
        epsilon_decay=0.99, epsilon_min=0.02
    )

    seed = 100  # Même seed de base que les expériences

    # ACO seul
    aco = ACOSolver(instance, ACOParams(
        num_ants=20, alpha=1.0, beta=3.0, rho=0.1, Q=100.0, max_iterations=150
    ), seed=seed)
    aco.solve()

    # ACO + Q-Learning
    hybrid = HybridACOQLSolver(instance, aco_params=ACOParams(
        num_ants=20, alpha=1.0, beta=3.0, rho=0.1, Q=100.0, max_iterations=150
    ), ql_params=ql_params, seed=seed)
    hybrid.solve()

    # Sauvegarder les données
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results")
    os.makedirs(output_dir, exist_ok=True)

    # Fichier dat pour pgfplots
    with open(os.path.join(output_dir, "convergence_aco.dat"), 'w') as f:
        f.write("iteration best_cost\n")
        for entry in aco.history:
            f.write(f"{entry['iteration']} {entry['best_cost']:.2f}\n")

    with open(os.path.join(output_dir, "convergence_hybrid.dat"), 'w') as f:
        f.write("iteration best_cost\n")
        for entry in hybrid.history:
            f.write(f"{entry['iteration']} {entry['best_cost']:.2f}\n")

    print(f"ACO final best:    {aco.best_cost:.2f}")
    print(f"Hybrid final best: {hybrid.best_cost:.2f}")
    print(f"Données sauvegardées dans {output_dir}/convergence_*.dat")


if __name__ == "__main__":
    main()
