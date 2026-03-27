"""
Q-Learning Reinforcement Learning Agent for Traffic Light Control.

State:  (queue_level, wait_level, current_phase)
        - queue_level:  0=Low, 1=Medium, 2=High, 3=VeryHigh
        - wait_level:   0=Short, 1=Medium, 2=Long, 3=VeryLong
        - current_phase: 0 or 1 (which direction has green)

Action: green_time index → maps to discrete durations
        0=15s, 1=25s, 2=35s, 3=50s, 4=65s

Reward: -(avg_waiting_time + queue_penalty + switch_penalty)
        Lower total wait = higher reward
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


# Discrete action space: green durations in seconds
ACTION_DURATIONS = [15, 25, 35, 50, 65]

# Discretization thresholds
QUEUE_BINS = [0, 8, 20, 35]       # → levels 0-3
WAIT_BINS = [0, 20, 50, 100]      # → levels 0-3
NUM_PHASES = 2                      # 2-phase intersection


def discretize(value: float, bins: list[float]) -> int:
    """Bin a continuous value into a discrete level."""
    for i in range(len(bins) - 1, -1, -1):
        if value >= bins[i]:
            return i
    return 0


@dataclass
class RLState:
    queue_level: int
    wait_level: int
    phase: int

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.queue_level, self.wait_level, self.phase)


@dataclass
class RLExperience:
    """Single experience tuple (s, a, r, s')."""
    state: tuple
    action: int
    reward: float
    next_state: tuple
    done: bool = False


class RLAgent:
    """
    Tabular Q-Learning agent for traffic light timing.

    The agent learns which green_time duration to choose for each
    (queue_level, wait_level, phase) state.
    """

    def __init__(
        self,
        alpha: float = 0.1,        # learning rate
        gamma: float = 0.95,       # discount factor
        epsilon: float = 0.15,     # exploration rate
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_actions = len(ACTION_DURATIONS)

        # Q-table: dict mapping state_tuple → array of Q-values per action
        self._q: dict[tuple, np.ndarray] = {}

        # Training stats
        self.total_steps: int = 0
        self.total_episodes: int = 0
        self.avg_reward: float = 0.0
        self._reward_history: list[float] = []

    def _get_q(self, state: tuple) -> np.ndarray:
        if state not in self._q:
            self._q[state] = np.zeros(self.n_actions)
        return self._q[state]

    def make_state(self, queue_length: float, waiting_time: float, phase: int) -> RLState:
        return RLState(
            queue_level=discretize(queue_length, QUEUE_BINS),
            wait_level=discretize(waiting_time, WAIT_BINS),
            phase=phase % NUM_PHASES,
        )

    def select_action(self, state: RLState) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self._get_q(state.to_tuple())
        return int(np.argmax(q))

    def get_green_time(self, action: int) -> int:
        """Convert action index to actual green duration."""
        return ACTION_DURATIONS[min(action, self.n_actions - 1)]

    def best_action(self, state: RLState) -> int:
        """Greedy action (no exploration)."""
        q = self._get_q(state.to_tuple())
        return int(np.argmax(q))

    def update(self, exp: RLExperience) -> float:
        """
        Q-Learning update rule:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]

        Returns TD error.
        """
        q_sa = self._get_q(exp.state)
        q_next = self._get_q(exp.next_state)

        target = exp.reward
        if not exp.done:
            target += self.gamma * np.max(q_next)

        td_error = target - q_sa[exp.action]
        q_sa[exp.action] += self.alpha * td_error

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.total_steps += 1

        return float(td_error)

    def end_episode(self, total_reward: float) -> None:
        """Called at end of episode for bookkeeping."""
        self.total_episodes += 1
        self._reward_history.append(total_reward)
        # Running average of last 100 episodes
        window = self._reward_history[-100:]
        self.avg_reward = sum(window) / len(window)

    @staticmethod
    def compute_reward(
        avg_wait: float,
        total_queue: float,
        did_switch: bool,
        throughput: int = 0,
    ) -> float:
        """
        Reward function design:
        - Penalize long average wait time
        - Penalize large queues
        - Small penalty for phase switches (encourages stability)
        - Bonus for throughput (vehicles cleared)
        """
        reward = 0.0
        reward -= avg_wait * 0.1           # wait penalty
        reward -= total_queue * 0.05       # queue penalty
        if did_switch:
            reward -= 2.0                  # switch penalty
        reward += throughput * 0.5         # throughput bonus
        return reward

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "avg_reward": self.avg_reward,
            "q_table": {str(k): v.tolist() for k, v in self._q.items()},
        }
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.total_steps = data.get("total_steps", 0)
        self.total_episodes = data.get("total_episodes", 0)
        self.avg_reward = data.get("avg_reward", 0.0)
        for k, v in data.get("q_table", {}).items():
            # Convert string key "(0, 1, 0)" back to tuple
            key = tuple(int(x) for x in k.strip("()").split(","))
            self._q[key] = np.array(v)

    def get_q_table_stats(self) -> dict:
        """Summary of Q-table for API/debug."""
        if not self._q:
            return {"states_visited": 0, "avg_q": 0.0, "max_q": 0.0}
        all_q = np.concatenate(list(self._q.values()))
        return {
            "states_visited": len(self._q),
            "avg_q": float(np.mean(all_q)),
            "max_q": float(np.max(all_q)),
            "min_q": float(np.min(all_q)),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "epsilon": round(self.epsilon, 4),
            "avg_reward": round(self.avg_reward, 2),
        }
