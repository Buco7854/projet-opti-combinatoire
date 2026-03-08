"""
Q-Learning pour l'ajustement dynamique des paramètres ACO

Le Q-Learning est un algorithme d'apprentissage par renforcement tabulaire qui
apprend une politique optimale pour ajuster les paramètres de l'ACO au cours
de la résolution.

États : discrétisation de l'état de la recherche
  - taux d'amélioration récent (stagnation vs progression)
  - diversité des solutions (convergence vs exploration)
  - phase de la recherche (début, milieu, fin)

Actions : stratégies de paramétrage ACO
  - exploitation : alpha élevé, beta modéré, rho faible (suivre les phéromones)
  - exploration : alpha faible, beta élevé, rho élevé (explorer, oublier)
  - balanced : paramètres équilibrés
  - intensify : renforcer les phéromones (rho très faible)
  - diversify : forte évaporation pour relancer la recherche
  - local_search : appliquer un 2-opt sur les meilleures solutions

Récompense : amélioration relative du meilleur coût
"""

import random
import numpy as np
from dataclasses import dataclass


@dataclass
class QLParams:
    """Paramètres du Q-Learning."""
    learning_rate: float = 0.2      # alpha (taux d'apprentissage)
    discount_factor: float = 0.95   # gamma
    epsilon: float = 0.15           # exploration initiale
    epsilon_decay: float = 0.99     # décroissance de epsilon
    epsilon_min: float = 0.02       # epsilon minimal

    # Discrétisation des états
    improvement_bins: int = 3       # [stagnation, faible, forte]
    diversity_bins: int = 3         # [convergé, moyen, diversifié]
    phase_bins: int = 3             # [début, milieu, fin]


# Stratégies prédéfinies (alpha, beta, rho)
STRATEGY_PRESETS = {
    'exploitation':  (2.0, 2.5, 0.05),
    'exploration':   (0.5, 4.0, 0.20),
    'balanced':      (1.0, 3.0, 0.10),
    'intensify':     (2.5, 2.0, 0.02),
    'diversify':     (0.3, 5.0, 0.30),
    'local_search':  (1.0, 3.0, 0.10),  # paramètres balanced + déclenchement 2-opt
}

ACTIONS = list(STRATEGY_PRESETS.keys())


class QLearningAgent:
    """Agent Q-Learning pour l'ajustement des paramètres ACO."""

    def __init__(self, params: QLParams = None, seed: int = None):
        self.params = params or QLParams()
        self.epsilon = self.params.epsilon

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Table Q : états x actions
        num_states = (self.params.improvement_bins *
                      self.params.diversity_bins *
                      self.params.phase_bins)
        self.num_actions = len(ACTIONS)
        self.q_table = np.zeros((num_states, self.num_actions))

        # Historique
        self.rewards_history = []
        self.actions_history = []
        self.states_history = []

    def _discretize_improvement(self, improvement_rate: float) -> int:
        """Discrétise le taux d'amélioration."""
        if improvement_rate <= 0.001:
            return 0  # stagnation
        elif improvement_rate <= 0.01:
            return 1  # faible amélioration
        else:
            return 2  # forte amélioration

    def _discretize_diversity(self, diversity: float) -> int:
        """Discrétise la diversité des solutions."""
        if diversity <= 0.05:
            return 0  # convergé
        elif diversity <= 0.15:
            return 1  # moyen
        else:
            return 2  # diversifié

    def _discretize_phase(self, progress: float) -> int:
        """Discrétise la phase de recherche (0.0 = début, 1.0 = fin)."""
        if progress <= 0.33:
            return 0  # début
        elif progress <= 0.66:
            return 1  # milieu
        else:
            return 2  # fin

    def get_state(self, improvement_rate: float, diversity: float,
                  progress: float) -> int:
        """Convertit les observations en un indice d'état discret."""
        imp = self._discretize_improvement(improvement_rate)
        div = self._discretize_diversity(diversity)
        phase = self._discretize_phase(progress)

        state = (imp * self.params.diversity_bins * self.params.phase_bins +
                 div * self.params.phase_bins + phase)
        return state

    def choose_action(self, state: int) -> int:
        """Politique epsilon-greedy pour choisir une action."""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Mise à jour de la table Q."""
        lr = self.params.learning_rate
        gamma = self.params.discount_factor

        best_next = np.max(self.q_table[next_state])
        td_target = reward + gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += lr * td_error

        # Décroissance de epsilon
        self.epsilon = max(self.params.epsilon_min,
                          self.epsilon * self.params.epsilon_decay)

        # Historique
        self.rewards_history.append(reward)
        self.actions_history.append(action)
        self.states_history.append(state)

    def apply_action(self, action_idx: int, current_alpha: float,
                     current_beta: float, current_rho: float):
        """Applique la stratégie choisie - retourne (alpha, beta, rho, apply_2opt)."""
        action_name = ACTIONS[action_idx]
        alpha, beta, rho = STRATEGY_PRESETS[action_name]
        apply_2opt = (action_name == 'local_search')
        return alpha, beta, rho, apply_2opt

    def get_action_name(self, action_idx: int) -> str:
        return ACTIONS[action_idx]

    def get_q_table_stats(self):
        """Retourne des statistiques sur la table Q."""
        return {
            'mean': float(np.mean(self.q_table)),
            'max': float(np.max(self.q_table)),
            'min': float(np.min(self.q_table)),
            'nonzero': int(np.count_nonzero(self.q_table)),
            'total_entries': self.q_table.size
        }
