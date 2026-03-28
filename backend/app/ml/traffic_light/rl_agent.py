"""
Deep Q-Network (DQN) Agent for Traffic Light Control.

Upgraded from tabular Q-Learning → DQN with:
  - Neural network (3-layer MLP) instead of Q-table
  - Experience Replay buffer (breaks temporal correlation)
  - Target network (stable training)
  - Continuous state input (no discretization needed)

State:  [queue_green, queue_red, avg_wait_green, avg_wait_red, phase_onehot(2)]
        = 6-dimensional continuous vector

Action: green_time index → maps to discrete durations
        0=15s, 1=25s, 2=35s, 3=50s, 4=65s

Reward: -(avg_wait * 0.1 + queue * 0.05 + switch * 2.0) + throughput * 0.5

Reference: Traffic_RL (SUMO-based DQN) by train_RL.py
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ── Constants ─────────────────────────────────────────────────────────────────

ACTION_DURATIONS = [15, 25, 35, 50, 65]
NUM_ACTIONS = len(ACTION_DURATIONS)
STATE_DIM = 6  # queue_g, queue_r, wait_g, wait_r, phase0, phase1
NUM_PHASES = 2

# Discretization thresholds (kept for backward compat & fallback)
QUEUE_BINS = [0, 8, 20, 35]
WAIT_BINS = [0, 20, 50, 100]


def discretize(value: float, bins: list[float]) -> int:
    for i in range(len(bins) - 1, -1, -1):
        if value >= bins[i]:
            return i
    return 0


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class RLState:
    queue_level: int
    wait_level: int
    phase: int

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.queue_level, self.wait_level, self.phase)


@dataclass
class RLExperience:
    state: tuple | list | np.ndarray
    action: int
    reward: float
    next_state: tuple | list | np.ndarray
    done: bool = False


# ── DQN Network ──────────────────────────────────────────────────────────────

if HAS_TORCH:
    class DQNNetwork(nn.Module):
        """3-layer MLP for Q-value estimation. Ref: Traffic_RL Model class."""

        def __init__(self, input_dims: int = STATE_DIM, fc1: int = 256,
                     fc2: int = 256, n_actions: int = NUM_ACTIONS):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dims, fc1),
                nn.ReLU(),
                nn.Linear(fc1, fc2),
                nn.ReLU(),
                nn.Linear(fc2, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int = 100_000):
        self._buf: deque = deque(maxlen=capacity)

    def push(self, exp: RLExperience) -> None:
        self._buf.append(exp)

    def sample(self, batch_size: int) -> list[RLExperience]:
        indices = np.random.choice(len(self._buf), batch_size, replace=False)
        return [self._buf[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buf)


# ── RLAgent (DQN) ────────────────────────────────────────────────────────────

class RLAgent:
    """
    DQN agent for traffic light timing.

    When PyTorch is available → full DQN with replay & target network.
    When PyTorch is missing → falls back to tabular Q-Learning.
    """

    def __init__(
        self,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 5e-4,    # linear decay per step (like Traffic_RL)
        batch_size: int = 64,
        replay_size: int = 100_000,
        target_update: int = 100,        # update target net every N steps
    ) -> None:
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update

        self.n_actions = NUM_ACTIONS
        self.total_steps: int = 0
        self.total_episodes: int = 0
        self.avg_reward: float = 0.0
        self._reward_history: list[float] = []

        self._use_dqn = HAS_TORCH

        if self._use_dqn:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_eval = DQNNetwork().to(self.device)
            self.q_target = DQNNetwork().to(self.device)
            self.q_target.load_state_dict(self.q_eval.state_dict())
            self.q_target.eval()
            self.optimizer = optim.Adam(self.q_eval.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()
            self.replay = ReplayBuffer(replay_size)
        else:
            # Fallback: tabular Q-table
            self._q: dict[tuple, np.ndarray] = {}
            self.alpha = 0.1

    # ── State Encoding ───────────────────────────────────────────────────────

    def encode_state(self, queue_green: float, queue_red: float,
                     wait_green: float, wait_red: float, phase: int) -> np.ndarray:
        """Encode state as continuous 6D vector (normalized)."""
        phase_oh = [1.0, 0.0] if phase == 0 else [0.0, 1.0]
        return np.array([
            queue_green / 45.0,   # normalize to ~[0, 1]
            queue_red / 45.0,
            wait_green / 120.0,
            wait_red / 120.0,
            phase_oh[0],
            phase_oh[1],
        ], dtype=np.float32)

    def make_state(self, queue_length: float, waiting_time: float, phase: int) -> RLState:
        """Backward compat: create discrete state (used by service layer)."""
        return RLState(
            queue_level=discretize(queue_length, QUEUE_BINS),
            wait_level=discretize(waiting_time, WAIT_BINS),
            phase=phase % NUM_PHASES,
        )

    def make_state_vec(self, queue_green: float, queue_red: float,
                       wait_green: float, wait_red: float, phase: int) -> np.ndarray:
        """Create continuous state vector for DQN."""
        return self.encode_state(queue_green, queue_red, wait_green, wait_red, phase)

    # ── Action Selection ─────────────────────────────────────────────────────

    def select_action(self, state: RLState | np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        if self._use_dqn:
            vec = self._state_to_vec(state)
            with torch.no_grad():
                t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_vals = self.q_eval(t)
            return int(torch.argmax(q_vals).item())
        else:
            key = state.to_tuple() if isinstance(state, RLState) else tuple(state)
            return int(np.argmax(self._get_q(key)))

    def best_action(self, state: RLState | np.ndarray) -> int:
        """Greedy action (no exploration)."""
        if self._use_dqn:
            vec = self._state_to_vec(state)
            with torch.no_grad():
                t = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_vals = self.q_eval(t)
            return int(torch.argmax(q_vals).item())
        else:
            key = state.to_tuple() if isinstance(state, RLState) else tuple(state)
            return int(np.argmax(self._get_q(key)))

    def get_green_time(self, action: int) -> int:
        return ACTION_DURATIONS[min(action, self.n_actions - 1)]

    # ── Learning ─────────────────────────────────────────────────────────────

    def update(self, exp: RLExperience) -> float:
        """
        Store experience and learn.

        DQN: add to replay buffer, sample batch, gradient step.
        Tabular: direct Q-update.
        """
        if self._use_dqn:
            return self._dqn_update(exp)
        else:
            return self._tabular_update(exp)

    def _dqn_update(self, exp: RLExperience) -> float:
        """DQN update with experience replay (ref: Traffic_RL train_RL.py)."""
        # Convert states to vectors
        s_vec = self._state_to_vec(exp.state)
        ns_vec = self._state_to_vec(exp.next_state)

        self.replay.push(RLExperience(
            state=s_vec, action=exp.action, reward=exp.reward,
            next_state=ns_vec, done=exp.done,
        ))

        self.total_steps += 1

        # Decay epsilon linearly (like Traffic_RL)
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # Don't learn until enough samples
        if len(self.replay) < self.batch_size:
            return 0.0

        # Sample batch
        batch = self.replay.sample(self.batch_size)

        states = torch.tensor(np.array([e.state for e in batch]),
                              dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array([e.next_state for e in batch]),
                                   dtype=torch.float32).to(self.device)
        actions = torch.tensor([e.action for e in batch],
                               dtype=torch.long).to(self.device)
        rewards = torch.tensor([e.reward for e in batch],
                               dtype=torch.float32).to(self.device)
        dones = torch.tensor([e.done for e in batch],
                             dtype=torch.bool).to(self.device)

        # Q(s, a) from eval network
        q_eval = self.q_eval(states)
        q_pred = q_eval.gather(1, actions.unsqueeze(1)).squeeze(1)

        # max Q(s', a') from target network
        with torch.no_grad():
            q_next = self.q_target(next_states)
            q_next_max = q_next.max(dim=1)[0]
            q_next_max[dones] = 0.0

        q_target = rewards + self.gamma * q_next_max

        # Gradient step
        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        return float(loss.item())

    def _tabular_update(self, exp: RLExperience) -> float:
        """Tabular Q-Learning fallback."""
        key = tuple(exp.state) if not isinstance(exp.state, tuple) else exp.state
        nkey = tuple(exp.next_state) if not isinstance(exp.next_state, tuple) else exp.next_state

        q_sa = self._get_q(key)
        q_next = self._get_q(nkey)

        target = exp.reward
        if not exp.done:
            target += self.gamma * np.max(q_next)

        td_error = target - q_sa[exp.action]
        q_sa[exp.action] += self.alpha * td_error

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        self.total_steps += 1
        return float(td_error)

    def end_episode(self, total_reward: float) -> None:
        self.total_episodes += 1
        self._reward_history.append(total_reward)
        window = self._reward_history[-100:]
        self.avg_reward = sum(window) / len(window)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _state_to_vec(self, state) -> np.ndarray:
        """Convert any state representation to a float32 numpy vector."""
        if isinstance(state, np.ndarray) and state.dtype == np.float32:
            return state
        if isinstance(state, RLState):
            # Convert discrete state → approximate continuous
            return self.encode_state(
                state.queue_level * 10, 0,
                state.wait_level * 25, 0,
                state.phase,
            )
        if isinstance(state, (tuple, list)):
            if len(state) == STATE_DIM:
                return np.array(state, dtype=np.float32)
            # 3-tuple from old RLState
            ql, wl, p = state[0], state[1], state[2]
            return self.encode_state(ql * 10, 0, wl * 25, 0, p)
        return np.zeros(STATE_DIM, dtype=np.float32)

    def _get_q(self, state: tuple) -> np.ndarray:
        """Tabular fallback Q-table."""
        if not hasattr(self, '_q'):
            self._q = {}
        if state not in self._q:
            self._q[state] = np.zeros(self.n_actions)
        return self._q[state]

    @staticmethod
    def compute_reward(
        avg_wait: float,
        total_queue: float,
        did_switch: bool,
        throughput: int = 0,
    ) -> float:
        reward = 0.0
        reward -= avg_wait * 0.1
        reward -= total_queue * 0.05
        if did_switch:
            reward -= 2.0
        reward += throughput * 0.5
        return reward

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_q_table_stats(self) -> dict:
        stats = {
            "states_visited": 0,
            "avg_q": 0.0,
            "max_q": 0.0,
            "min_q": 0.0,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "epsilon": round(self.epsilon, 4),
            "avg_reward": round(self.avg_reward, 2),
            "backend": "DQN" if self._use_dqn else "Q-table",
        }

        if self._use_dqn and HAS_TORCH:
            # Sample some states to estimate Q stats
            try:
                test_states = torch.randn(100, STATE_DIM).to(self.device)
                with torch.no_grad():
                    q_vals = self.q_eval(test_states)
                stats["avg_q"] = float(q_vals.mean().item())
                stats["max_q"] = float(q_vals.max().item())
                stats["min_q"] = float(q_vals.min().item())
                stats["states_visited"] = self.total_steps
            except Exception:
                pass
        elif hasattr(self, '_q') and self._q:
            all_q = np.concatenate(list(self._q.values()))
            stats["states_visited"] = len(self._q)
            stats["avg_q"] = float(np.mean(all_q))
            stats["max_q"] = float(np.max(all_q))
            stats["min_q"] = float(np.min(all_q))

        return stats

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "avg_reward": self.avg_reward,
            "use_dqn": self._use_dqn,
        }

        if self._use_dqn and HAS_TORCH:
            torch.save({
                "q_eval": self.q_eval.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "meta": meta,
            }, str(path.with_suffix(".pt")))
            # Also save JSON meta for API
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        else:
            # Tabular fallback
            meta["q_table"] = {str(k): v.tolist() for k, v in self._q.items()}
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    def load(self, path: str | Path) -> None:
        path = Path(path)

        # Try loading DQN checkpoint first
        pt_path = path.with_suffix(".pt")
        if self._use_dqn and HAS_TORCH and pt_path.exists():
            checkpoint = torch.load(str(pt_path), map_location=self.device, weights_only=False)
            self.q_eval.load_state_dict(checkpoint["q_eval"])
            self.q_target.load_state_dict(checkpoint["q_target"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            meta = checkpoint.get("meta", {})
            self.epsilon = meta.get("epsilon", self.epsilon)
            self.total_steps = meta.get("total_steps", 0)
            self.total_episodes = meta.get("total_episodes", 0)
            self.avg_reward = meta.get("avg_reward", 0.0)
            return

        # Fallback: load JSON (tabular or meta only)
        json_path = path.with_suffix(".json")
        if json_path.exists():
            data = json.loads(json_path.read_text())
            self.epsilon = data.get("epsilon", self.epsilon)
            self.total_steps = data.get("total_steps", 0)
            self.total_episodes = data.get("total_episodes", 0)
            self.avg_reward = data.get("avg_reward", 0.0)
            if not self._use_dqn and "q_table" in data:
                self._q = {}
                for k, v in data["q_table"].items():
                    key = tuple(int(x) for x in k.strip("()").split(","))
                    self._q[key] = np.array(v)
