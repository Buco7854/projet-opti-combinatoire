"""
Interface web Flask pour la démonstration du projet CVRP ACO + Q-Learning.
"""

import json
import os
import time
from flask import Flask, render_template, request, jsonify
from src.cvrp import CVRPInstance, generate_random_instance
from src.aco import ACOSolver, ACOParams
from src.hybrid_aco_ql import HybridACOQLSolver
from src.qlearning import QLParams

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/solve', methods=['POST'])
def solve():
    """Lance la résolution et retourne les résultats."""
    data = request.json

    num_customers = int(data.get('num_customers', 20))
    capacity = int(data.get('capacity', 80))
    num_ants = int(data.get('num_ants', 20))
    max_iterations = int(data.get('max_iterations', 80))
    seed = int(data.get('seed', 42))

    # Générer l'instance
    instance = generate_random_instance(
        num_customers=num_customers,
        capacity=capacity,
        max_coord=100,
        max_demand=25,
        seed=seed
    )

    aco_params = ACOParams(
        num_ants=num_ants,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        Q=100.0,
        max_iterations=max_iterations
    )

    ql_params = QLParams(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.05
    )

    # --- ACO seul ---
    aco = ACOSolver(instance, ACOParams(
        num_ants=num_ants, alpha=1.0, beta=3.0, rho=0.1, Q=100.0,
        max_iterations=max_iterations
    ), seed=seed)
    t0 = time.time()
    sol_aco = aco.solve()
    time_aco = time.time() - t0

    # --- Hybride ---
    hybrid = HybridACOQLSolver(
        instance,
        aco_params=ACOParams(
            num_ants=num_ants, alpha=1.0, beta=3.0, rho=0.1, Q=100.0,
            max_iterations=max_iterations
        ),
        ql_params=ql_params,
        seed=seed
    )
    t0 = time.time()
    sol_hybrid = hybrid.solve()
    time_hybrid = time.time() - t0

    # Préparer les données de réponse
    instance_data = {
        'depot': list(instance.depot),
        'customers': [list(c) for c in instance.customers],
        'demands': instance.demands,
        'capacity': instance.vehicle_capacity
    }

    def format_solution(sol):
        routes = []
        for route in sol.routes:
            points = [list(instance.depot)]
            for c in route:
                points.append(list(instance.customers[c - 1]))
            points.append(list(instance.depot))
            routes.append({
                'customers': route,
                'points': points,
                'load': sum(instance.demands[c - 1] for c in route)
            })
        return routes

    # Sous-échantillonner l'historique pour ne pas surcharger le frontend
    step = max(1, max_iterations // 100)
    aco_hist = [h for i, h in enumerate(aco.history) if i % step == 0 or i == len(aco.history) - 1]
    hybrid_hist = [h for i, h in enumerate(hybrid.history) if i % step == 0 or i == len(hybrid.history) - 1]

    # Param history sous-échantillonné
    param_hist = [h for i, h in enumerate(hybrid.param_history) if i % step == 0 or i == len(hybrid.param_history) - 1]

    response = {
        'instance': instance_data,
        'aco': {
            'cost': sol_aco.cost,
            'time': time_aco,
            'routes': format_solution(sol_aco),
            'num_routes': len(sol_aco.routes),
            'history': aco_hist
        },
        'hybrid': {
            'cost': sol_hybrid.cost,
            'time': time_hybrid,
            'routes': format_solution(sol_hybrid),
            'num_routes': len(sol_hybrid.routes),
            'history': hybrid_hist,
            'param_history': param_hist,
            'ql_stats': hybrid.ql_agent.get_q_table_stats()
        },
        'improvement_pct': ((sol_aco.cost - sol_hybrid.cost) / sol_aco.cost * 100)
            if sol_aco.cost > 0 else 0
    }

    return jsonify(response)


@app.route('/api/experiments', methods=['GET'])
def get_experiments():
    """Retourne les résultats d'expériences pré-calculées."""
    results_path = os.path.join('results', 'results_summary.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
