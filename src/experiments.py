"""
Module d'expérimentation : comparaison ACO vs ACO+Q-Learning.

Lance les deux approches sur plusieurs instances et collecte les résultats
pour l'analyse comparative.
"""

import json
import time
import os
import numpy as np
from src.cvrp import CVRPInstance, CVRPSolution, generate_random_instance
from src.aco import ACOSolver, ACOParams
from src.hybrid_aco_ql import HybridACOQLSolver
from src.qlearning import QLParams


def create_benchmark_instances() -> list:
    """Crée un ensemble d'instances benchmark de tailles variées."""
    instances = []

    configs = [
        {"n": 10, "cap": 50, "name": "small_10", "seed": 42},
        {"n": 15, "cap": 60, "name": "small_15", "seed": 43},
        {"n": 20, "cap": 80, "name": "medium_20", "seed": 44},
        {"n": 25, "cap": 80, "name": "medium_25", "seed": 45},
        {"n": 30, "cap": 100, "name": "medium_30", "seed": 46},
        {"n": 40, "cap": 120, "name": "large_40", "seed": 47},
        {"n": 50, "cap": 150, "name": "large_50", "seed": 48},
    ]

    for cfg in configs:
        inst = generate_random_instance(
            num_customers=cfg["n"],
            capacity=cfg["cap"],
            max_coord=100,
            max_demand=25,
            seed=cfg["seed"]
        )
        inst.name = cfg["name"]
        instances.append(inst)

    return instances


def run_single_experiment(instance: CVRPInstance, aco_params: ACOParams,
                          ql_params: QLParams, num_runs: int = 5,
                          seed_base: int = 100) -> dict:
    """Exécute ACO et ACO+QL sur une instance, plusieurs runs."""
    aco_results = []
    hybrid_results = []

    for run in range(num_runs):
        seed = seed_base + run

        # ACO seul
        aco = ACOSolver(instance, ACOParams(
            num_ants=aco_params.num_ants,
            alpha=aco_params.alpha,
            beta=aco_params.beta,
            rho=aco_params.rho,
            Q=aco_params.Q,
            max_iterations=aco_params.max_iterations
        ), seed=seed)
        t0 = time.time()
        sol_aco = aco.solve()
        t_aco = time.time() - t0

        aco_results.append({
            'cost': sol_aco.cost,
            'time': t_aco,
            'num_routes': len(sol_aco.routes),
            'history': aco.history
        })

        # ACO + Q-Learning
        hybrid = HybridACOQLSolver(
            instance,
            aco_params=ACOParams(
                num_ants=aco_params.num_ants,
                alpha=aco_params.alpha,
                beta=aco_params.beta,
                rho=aco_params.rho,
                Q=aco_params.Q,
                max_iterations=aco_params.max_iterations
            ),
            ql_params=ql_params,
            seed=seed
        )
        t0 = time.time()
        sol_hybrid = hybrid.solve()
        t_hybrid = time.time() - t0

        hybrid_results.append({
            'cost': sol_hybrid.cost,
            'time': t_hybrid,
            'num_routes': len(sol_hybrid.routes),
            'history': hybrid.history,
            'param_history': hybrid.param_history,
            'ql_stats': hybrid.ql_agent.get_q_table_stats()
        })

    return {
        'instance_name': instance.name,
        'num_customers': instance.num_customers,
        'aco': {
            'costs': [r['cost'] for r in aco_results],
            'times': [r['time'] for r in aco_results],
            'best_cost': min(r['cost'] for r in aco_results),
            'avg_cost': np.mean([r['cost'] for r in aco_results]),
            'std_cost': np.std([r['cost'] for r in aco_results]),
            'avg_time': np.mean([r['time'] for r in aco_results]),
        },
        'hybrid': {
            'costs': [r['cost'] for r in hybrid_results],
            'times': [r['time'] for r in hybrid_results],
            'best_cost': min(r['cost'] for r in hybrid_results),
            'avg_cost': np.mean([r['cost'] for r in hybrid_results]),
            'std_cost': np.std([r['cost'] for r in hybrid_results]),
            'avg_time': np.mean([r['time'] for r in hybrid_results]),
        },
        'improvement_pct': ((np.mean([r['cost'] for r in aco_results]) -
                             np.mean([r['cost'] for r in hybrid_results])) /
                            np.mean([r['cost'] for r in aco_results]) * 100),
        'details': {
            'aco_runs': aco_results,
            'hybrid_runs': hybrid_results
        }
    }


def run_all_experiments(output_dir: str = "results", num_runs: int = 5) -> list:
    """Lance toutes les expériences et sauvegarde les résultats."""
    os.makedirs(output_dir, exist_ok=True)

    instances = create_benchmark_instances()

    aco_params = ACOParams(
        num_ants=20,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        Q=100.0,
        max_iterations=150
    )

    ql_params = QLParams(
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=0.15,
        epsilon_decay=0.99,
        epsilon_min=0.02
    )

    all_results = []

    for inst in instances:
        print(f"Instance: {inst.name} ({inst.num_customers} clients)...")
        result = run_single_experiment(inst, aco_params, ql_params,
                                       num_runs=num_runs)
        all_results.append(result)

        print(f"  ACO:    best={result['aco']['best_cost']:.2f}, "
              f"avg={result['aco']['avg_cost']:.2f} ± {result['aco']['std_cost']:.2f}")
        print(f"  Hybrid: best={result['hybrid']['best_cost']:.2f}, "
              f"avg={result['hybrid']['avg_cost']:.2f} ± {result['hybrid']['std_cost']:.2f}")
        print(f"  Amélioration: {result['improvement_pct']:.2f}%")

    # Sauvegarder les résultats (sans les historiques détaillés pour le JSON)
    summary = []
    for r in all_results:
        summary.append({
            'instance_name': r['instance_name'],
            'num_customers': r['num_customers'],
            'aco_best': r['aco']['best_cost'],
            'aco_avg': float(r['aco']['avg_cost']),
            'aco_std': float(r['aco']['std_cost']),
            'aco_time': float(r['aco']['avg_time']),
            'hybrid_best': r['hybrid']['best_cost'],
            'hybrid_avg': float(r['hybrid']['avg_cost']),
            'hybrid_std': float(r['hybrid']['std_cost']),
            'hybrid_time': float(r['hybrid']['avg_time']),
            'improvement_pct': float(r['improvement_pct'])
        })

    with open(os.path.join(output_dir, "results_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nRésultats sauvegardés dans {output_dir}/results_summary.json")
    return all_results


if __name__ == "__main__":
    run_all_experiments()
