"""
ACO - Ant Colony Optimization pour le CVRP

Implémentation classique de l'algorithme de colonies de fourmis :
- Chaque fourmi construit une solution complète
- Les phéromones guident la construction
- Évaporation + renforcement des meilleurs chemins

Paramètres principaux :
- alpha : importance des phéromones
- beta : importance de l'heuristique (1/distance)
- rho : taux d'évaporation
- Q : facteur de dépôt de phéromone
- num_ants : nombre de fourmis par itération
"""

import random
import time
from dataclasses import dataclass, field
from src.cvrp import CVRPInstance, CVRPSolution


@dataclass
class ACOParams:
    """Paramètres de l'algorithme ACO."""
    num_ants: int = 20
    alpha: float = 1.0       # poids des phéromones
    beta: float = 3.0        # poids de l'heuristique
    rho: float = 0.1         # taux d'évaporation
    Q: float = 100.0         # facteur de dépôt
    tau_min: float = 0.1     # phéromone minimale
    tau_max: float = 10.0    # phéromone maximale
    max_iterations: int = 100


class ACOSolver:
    """Solveur ACO classique pour le CVRP."""

    def __init__(self, instance: CVRPInstance, params: ACOParams = None, seed: int = None):
        self.instance = instance
        self.params = params or ACOParams()
        self.n = instance.num_customers + 1  # dépôt + clients
        self.dist = instance.distance_matrix
        if seed is not None:
            random.seed(seed)

        # Initialisation des phéromones
        tau_init = 1.0
        self.pheromone = [[tau_init] * self.n for _ in range(self.n)]

        # Heuristique : 1/distance (eta)
        self.eta = [[0.0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.dist[i][j] > 0:
                    self.eta[i][j] = 1.0 / self.dist[i][j]

        # Historique pour analyse
        self.history = []
        self.best_solution = None
        self.best_cost = float('inf')

    def _construct_solution(self) -> CVRPSolution:
        """Construction d'une solution par une fourmi."""
        unvisited = set(range(1, self.n))
        routes = []

        while unvisited:
            route = []
            current = 0
            load = 0

            while unvisited:
                # Calculer les probabilités de transition
                candidates = []
                probs = []
                for j in unvisited:
                    demand_j = self.instance.demands[j - 1]
                    if load + demand_j <= self.instance.vehicle_capacity:
                        tau = self.pheromone[current][j] ** self.params.alpha
                        eta = self.eta[current][j] ** self.params.beta
                        score = tau * eta
                        candidates.append(j)
                        probs.append(score)

                if not candidates:
                    break

                # Sélection proportionnelle (roulette)
                total = sum(probs)
                if total == 0:
                    chosen = random.choice(candidates)
                else:
                    probs = [p / total for p in probs]
                    r = random.random()
                    cumsum = 0
                    chosen = candidates[-1]
                    for idx, p in enumerate(probs):
                        cumsum += p
                        if r <= cumsum:
                            chosen = candidates[idx]
                            break

                route.append(chosen)
                load += self.instance.demands[chosen - 1]
                unvisited.remove(chosen)
                current = chosen

            if route:
                routes.append(route)

        solution = CVRPSolution(routes=routes)
        solution.compute_cost(self.instance)
        return solution

    def _update_pheromone(self, solutions: list):
        """Mise à jour des phéromones : évaporation + dépôt."""
        p = self.params

        # Évaporation
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - p.rho)
                self.pheromone[i][j] = max(self.pheromone[i][j], p.tau_min)

        # Dépôt de phéromone par les meilleures fourmis (stratégie élitiste)
        sorted_sols = sorted(solutions, key=lambda s: s.cost)
        elite_count = max(1, len(sorted_sols) // 3)
        for rank, sol in enumerate(sorted_sols[:elite_count]):
            weight = (elite_count - rank) / elite_count  # meilleure fourmi = poids max
            deposit = weight * p.Q / sol.cost if sol.cost > 0 else 0
            for route in sol.routes:
                path = [0] + route + [0]
                for k in range(len(path) - 1):
                    i, j = path[k], path[k + 1]
                    self.pheromone[i][j] += deposit
                    self.pheromone[j][i] += deposit

        # Dépôt bonus par la meilleure solution globale
        if self.best_solution and self.best_cost < float('inf'):
            deposit = p.Q / self.best_cost * 0.5
            for route in self.best_solution.routes:
                path = [0] + route + [0]
                for k in range(len(path) - 1):
                    i, j = path[k], path[k + 1]
                    self.pheromone[i][j] += deposit
                    self.pheromone[j][i] += deposit

        # Borne supérieure
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] = min(self.pheromone[i][j], p.tau_max)

    def solve(self, callback=None) -> CVRPSolution:
        """Exécute l'algorithme ACO complet."""
        start_time = time.time()

        for iteration in range(self.params.max_iterations):
            # Construction des solutions
            solutions = []
            for _ in range(self.params.num_ants):
                sol = self._construct_solution()
                solutions.append(sol)

                if sol.cost < self.best_cost:
                    self.best_cost = sol.cost
                    self.best_solution = sol

            # Mise à jour des phéromones
            self._update_pheromone(solutions)

            # Enregistrer l'historique
            avg_cost = sum(s.cost for s in solutions) / len(solutions)
            elapsed = time.time() - start_time
            self.history.append({
                'iteration': iteration,
                'best_cost': self.best_cost,
                'avg_cost': avg_cost,
                'time': elapsed
            })

            if callback:
                callback(iteration, self.best_cost, avg_cost)

        return self.best_solution

    def get_pheromone_matrix(self):
        """Retourne la matrice de phéromones actuelle."""
        return [row[:] for row in self.pheromone]
