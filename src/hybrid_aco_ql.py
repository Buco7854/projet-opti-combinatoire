"""
Solveur hybride ACO + Q-Learning

Combine l'ACO classique avec un agent Q-Learning qui choisit dynamiquement
la stratégie de paramétrage de l'ACO à chaque itération en fonction
de l'état de la recherche.

Stratégies disponibles :
- exploitation : suivre les phéromones (alpha élevé)
- exploration : explorer de nouvelles routes (beta élevé, rho élevé)
- balanced : paramètres équilibrés
- intensify : renforcer les meilleures routes (rho très faible)
- diversify : relancer la recherche (forte évaporation)
- local_search : appliquer le 2-opt sur les meilleures solutions
"""

import time
import random
import numpy as np
from src.cvrp import CVRPInstance, CVRPSolution
from src.aco import ACOSolver, ACOParams
from src.qlearning import QLearningAgent, QLParams, ACTIONS
from src.local_search import improve_solution


class HybridACOQLSolver:
    """Solveur hybride : ACO avec ajustement par Q-Learning."""

    def __init__(self, instance: CVRPInstance, aco_params: ACOParams = None,
                 ql_params: QLParams = None, seed: int = None):
        self.instance = instance
        self.aco_params = aco_params or ACOParams()
        self.ql_params = ql_params or QLParams()

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.aco = ACOSolver(instance, self.aco_params, seed=seed)
        self.ql_agent = QLearningAgent(self.ql_params, seed=seed)

        self.history = []
        self.param_history = []
        self.best_solution = None
        self.best_cost = float('inf')

    def _compute_diversity(self, solutions: list) -> float:
        """Mesure la diversité des solutions (coefficient de variation des coûts)."""
        costs = [s.cost for s in solutions]
        if len(costs) < 2:
            return 0.0
        mean_cost = np.mean(costs)
        if mean_cost == 0:
            return 0.0
        return float(np.std(costs) / mean_cost)

    def _compute_improvement_rate(self, prev_best: float, curr_best: float) -> float:
        """Calcule le taux d'amélioration relative."""
        if prev_best == 0 or prev_best == float('inf'):
            return 0.0
        return max(0, (prev_best - curr_best) / prev_best)

    def _compute_reward(self, prev_best: float, curr_best: float,
                        avg_cost: float, prev_avg: float) -> float:
        """Calcule la récompense pour l'agent QL."""
        if prev_best == float('inf'):
            return 0.0

        best_improvement = (prev_best - curr_best) / prev_best
        avg_improvement = (prev_avg - avg_cost) / prev_avg if prev_avg > 0 else 0

        if best_improvement > 0.001:
            reward = 1.0 + 10.0 * best_improvement
        elif avg_improvement > 0:
            reward = 0.5 * avg_improvement
        else:
            reward = -0.05

        return reward

    def solve(self, callback=None) -> CVRPSolution:
        """Exécute le solveur hybride ACO + Q-Learning."""
        start_time = time.time()
        max_iter = self.aco_params.max_iterations
        prev_best = float('inf')
        prev_avg = float('inf')
        warmup = max(5, max_iter // 10)

        for iteration in range(max_iter):
            progress = iteration / max_iter

            # 1. Observer l'état
            improvement_rate = self._compute_improvement_rate(prev_best, self.best_cost)
            if self.history:
                last_diversity = self.history[-1].get('diversity', 0.1)
            else:
                last_diversity = 0.1

            state = self.ql_agent.get_state(improvement_rate, last_diversity, progress)

            # 2. Choisir et appliquer une stratégie (après le warm-up)
            apply_2opt = False
            if iteration >= warmup:
                action = self.ql_agent.choose_action(state)
                new_alpha, new_beta, new_rho, apply_2opt = self.ql_agent.apply_action(
                    action,
                    self.aco.params.alpha,
                    self.aco.params.beta,
                    self.aco.params.rho
                )
                self.aco.params.alpha = new_alpha
                self.aco.params.beta = new_beta
                self.aco.params.rho = new_rho
            else:
                action = 2  # 'balanced' pendant le warm-up
                new_alpha = self.aco.params.alpha
                new_beta = self.aco.params.beta
                new_rho = self.aco.params.rho

            # Enregistrer les paramètres
            self.param_history.append({
                'iteration': iteration,
                'alpha': new_alpha,
                'beta': new_beta,
                'rho': new_rho,
                'action': self.ql_agent.get_action_name(action),
                'epsilon': self.ql_agent.epsilon
            })

            # 3. Exécuter une itération ACO
            prev_best = self.best_cost
            solutions = []
            for _ in range(self.aco.params.num_ants):
                sol = self.aco._construct_solution()
                # Appliquer 2-opt si la stratégie le demande
                if apply_2opt:
                    sol = improve_solution(sol, self.instance)
                solutions.append(sol)
                if sol.cost < self.best_cost:
                    self.best_cost = sol.cost
                    self.best_solution = sol

            self.aco._update_pheromone(solutions)

            # 4. Calculer métriques et récompense
            avg_cost = sum(s.cost for s in solutions) / len(solutions)
            diversity = self._compute_diversity(solutions)

            reward = self._compute_reward(prev_best, self.best_cost, avg_cost, prev_avg)
            prev_avg = avg_cost

            # 5. Observer le nouvel état et mettre à jour Q
            new_improvement = self._compute_improvement_rate(prev_best, self.best_cost)
            next_state = self.ql_agent.get_state(new_improvement, diversity, progress)

            if iteration >= warmup:
                self.ql_agent.update(state, action, reward, next_state)

            # Historique
            elapsed = time.time() - start_time
            self.history.append({
                'iteration': iteration,
                'best_cost': self.best_cost,
                'avg_cost': avg_cost,
                'diversity': diversity,
                'reward': reward,
                'time': elapsed
            })

            self.aco.history.append({
                'iteration': iteration,
                'best_cost': self.best_cost,
                'avg_cost': avg_cost,
                'time': elapsed
            })
            self.aco.best_cost = self.best_cost
            self.aco.best_solution = self.best_solution

            if callback:
                callback(iteration, self.best_cost, avg_cost)

        return self.best_solution

    def get_results_summary(self) -> dict:
        """Résumé des résultats pour l'analyse."""
        return {
            'best_cost': self.best_cost,
            'total_time': self.history[-1]['time'] if self.history else 0,
            'iterations': len(self.history),
            'final_params': self.param_history[-1] if self.param_history else {},
            'ql_stats': self.ql_agent.get_q_table_stats(),
            'total_rewards': sum(self.ql_agent.rewards_history),
            'avg_reward': (np.mean(self.ql_agent.rewards_history)
                          if self.ql_agent.rewards_history else 0),
        }
