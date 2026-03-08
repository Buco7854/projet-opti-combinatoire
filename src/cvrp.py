"""
CVRP - Capacitated Vehicle Routing Problem

Modélisation du problème :
- Un dépôt central (noeud 0)
- N clients avec des demandes spécifiques
- Une flotte de véhicules homogènes avec capacité maximale Q
- Objectif : minimiser la distance totale parcourue
- Contraintes : chaque client visité exactement une fois, capacité respectée
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class CVRPInstance:
    """Représente une instance du problème CVRP."""
    name: str
    num_customers: int
    vehicle_capacity: int
    depot: tuple  # (x, y)
    customers: list  # [(x, y), ...]
    demands: list  # [demand_i, ...]
    optimal_cost: float = None

    def distance(self, i: int, j: int) -> float:
        """Distance euclidienne entre deux noeuds (0 = dépôt)."""
        ci = self.depot if i == 0 else self.customers[i - 1]
        cj = self.depot if j == 0 else self.customers[j - 1]
        return math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)

    @property
    def distance_matrix(self) -> list:
        """Matrice de distances complète (dépôt + clients)."""
        n = self.num_customers + 1
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(i, j)
                matrix[i][j] = d
                matrix[j][i] = d
        return matrix

    def total_demand(self) -> int:
        return sum(self.demands)

    def min_vehicles(self) -> int:
        return math.ceil(self.total_demand() / self.vehicle_capacity)


@dataclass
class CVRPSolution:
    """Solution du CVRP : liste de routes."""
    routes: list  # [[client_ids], ...]
    cost: float = 0.0

    def is_valid(self, instance: CVRPInstance) -> bool:
        """Vérifie la validité de la solution."""
        visited = set()
        for route in self.routes:
            load = sum(instance.demands[c - 1] for c in route)
            if load > instance.vehicle_capacity:
                return False
            for c in route:
                if c in visited:
                    return False
                visited.add(c)
        return len(visited) == instance.num_customers

    def compute_cost(self, instance: CVRPInstance) -> float:
        """Calcule le coût total (distance) de la solution."""
        total = 0.0
        dist = instance.distance_matrix
        for route in self.routes:
            if not route:
                continue
            total += dist[0][route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i]][route[i + 1]]
            total += dist[route[-1]][0]
        self.cost = total
        return total


def generate_random_instance(num_customers: int, capacity: int,
                              max_coord: int = 100, max_demand: int = 30,
                              seed: int = None) -> CVRPInstance:
    """Génère une instance CVRP aléatoire."""
    if seed is not None:
        random.seed(seed)
    depot = (max_coord // 2, max_coord // 2)
    customers = [(random.randint(0, max_coord), random.randint(0, max_coord))
                 for _ in range(num_customers)]
    demands = [random.randint(1, max_demand) for _ in range(num_customers)]
    return CVRPInstance(
        name=f"random_{num_customers}",
        num_customers=num_customers,
        vehicle_capacity=capacity,
        depot=depot,
        customers=customers,
        demands=demands
    )


def parse_vrplib(filepath: str) -> CVRPInstance:
    """Parse un fichier au format VRPLIB (.vrp)."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    name = "unknown"
    capacity = 0
    dimension = 0
    coords = {}
    demands_dict = {}
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("NAME"):
            name = line.split(":")[-1].strip()
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[-1].strip())
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[-1].strip())
        elif line == "NODE_COORD_SECTION":
            section = "coords"
        elif line == "DEMAND_SECTION":
            section = "demands"
        elif line == "DEPOT_SECTION":
            section = "depot"
        elif line == "EOF":
            break
        elif section == "coords":
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                coords[idx] = (float(parts[1]), float(parts[2]))
        elif section == "demands":
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                demands_dict[idx] = int(parts[1])

    depot = coords.get(1, (0, 0))
    customers = [coords[i] for i in range(2, dimension + 1)]
    demands = [demands_dict.get(i, 0) for i in range(2, dimension + 1)]

    return CVRPInstance(
        name=name,
        num_customers=dimension - 1,
        vehicle_capacity=capacity,
        depot=depot,
        customers=customers,
        demands=demands
    )
