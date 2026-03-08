"""
Opérateurs de recherche locale pour le CVRP.

2-opt intra-route : inverse un segment dans une route pour réduire la distance.
"""

from src.cvrp import CVRPInstance, CVRPSolution


def two_opt_route(route: list, dist_matrix: list) -> list:
    """Applique le 2-opt sur une route unique (sans le dépôt)."""
    improved = True
    best = route[:]
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                # Calcul du gain du 2-opt swap
                # Noeuds : prev_i -> i -> ... -> j -> next_j
                node_i = best[i]
                node_j = best[j]
                prev_i = 0 if i == 0 else best[i - 1]
                next_j = 0 if j == len(best) - 1 else best[j + 1]

                old_cost = dist_matrix[prev_i][node_i] + dist_matrix[node_j][next_j]
                new_cost = dist_matrix[prev_i][node_j] + dist_matrix[node_i][next_j]

                if new_cost < old_cost - 1e-10:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return best


def improve_solution(solution: CVRPSolution, instance: CVRPInstance) -> CVRPSolution:
    """Applique le 2-opt sur chaque route de la solution."""
    dist = instance.distance_matrix
    new_routes = []
    for route in solution.routes:
        if len(route) >= 3:
            new_route = two_opt_route(route, dist)
            new_routes.append(new_route)
        else:
            new_routes.append(route[:])

    new_sol = CVRPSolution(routes=new_routes)
    new_sol.compute_cost(instance)
    return new_sol
