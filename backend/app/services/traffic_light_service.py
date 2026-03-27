"""
Traffic Light Service – orchestrates Fuzzy + RL + GA pipeline.

Connects to the existing vehicle counting system:
  - Uses vehicle counts from stream_service as real-time inputs
  - Fuzzy controller computes optimal green time
  - RL agent learns from outcomes
  - GA optimizes fuzzy params in background

Modes:
  - "manual":  fixed green times, no optimization
  - "fuzzy":   fuzzy controller decides green time
  - "rl":      RL agent selects action (green time)
  - "auto":    RL agent with fuzzy as fallback
"""

from __future__ import annotations

import threading
import time
import json
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger
from app.models.traffic_light_model import (
    TrafficLightState, TrafficLightPhase, FuzzyDecision,
    RLStatus, GAStatus, SimulationResult,
)
from app.ml.traffic_light.fuzzy_controller import FuzzyController
from app.ml.traffic_light.rl_agent import RLAgent, RLExperience, ACTION_DURATIONS
from app.ml.traffic_light.genetic_optimizer import GeneticOptimizer, GAConfig
from app.ml.traffic_light.simulator import TrafficSimulator


# Paths for persistence
DATA_DIR = Path("data") / "traffic_light"
RL_SAVE_PATH = DATA_DIR / "rl_agent.json"
FUZZY_SAVE_PATH = DATA_DIR / "fuzzy_params.json"


class TrafficLightService:
    """
    Singleton service managing the full traffic light optimization pipeline.
    """

    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Core components
        self._fuzzy = FuzzyController()
        self._rl = RLAgent()
        self._ga = GeneticOptimizer()
        self._sim = TrafficSimulator()

        # State
        self._state = TrafficLightState()
        self._mode = "fuzzy"  # default mode
        self._cycle_thread: threading.Thread | None = None
        self._cycle_running = False
        self._ga_thread: threading.Thread | None = None

        # Real-time inputs (updated from stream_service)
        self._queue_lengths: list[int] = [0, 0]     # per phase
        self._wait_times: list[float] = [0.0, 0.0]  # per phase
        self._throughput: list[int] = [0, 0]

        # RL episode tracking
        self._prev_rl_state: tuple | None = None
        self._prev_rl_action: int | None = None
        self._episode_reward: float = 0.0

        # Load saved state
        self._load_state()

    # ── Real-time Input ──────────────────────────────────────────────────────

    def update_from_counts(
        self,
        count_in: int,
        count_out: int,
        total: int,
        fps: float,
    ) -> None:
        """
        Called by stream_service each frame to update traffic state.
        Maps IN/OUT counts to 2-phase queue estimates.
        """
        # Phase 0 = IN direction, Phase 1 = OUT direction
        self._queue_lengths[0] = count_in
        self._queue_lengths[1] = count_out

        # Estimate wait time from queue and fps
        if fps > 0:
            self._wait_times[0] = count_in / max(fps, 0.1) * 10  # rough estimate
            self._wait_times[1] = count_out / max(fps, 0.1) * 10
        self._throughput = [count_in, count_out]

    # ── Mode Control ─────────────────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        if mode in ("manual", "fuzzy", "rl", "auto"):
            self._mode = mode
            self._state.mode = mode
            logger.info("TrafficLight: mode → %s", mode)

    # ── Fuzzy Decision ───────────────────────────────────────────────────────

    def get_fuzzy_decision(self, phase: int = 0) -> FuzzyDecision:
        """Get fuzzy controller recommendation for a phase."""
        q = self._queue_lengths[phase]
        w = self._wait_times[phase]
        green = self._fuzzy.decide(queue_length=q, waiting_time=w)
        return FuzzyDecision(
            green_time=round(green, 1),
            queue_input=q,
            wait_input=round(w, 1),
        )

    # ── RL Decision ──────────────────────────────────────────────────────────

    def get_rl_decision(self, phase: int = 0) -> dict:
        """Get RL agent recommendation."""
        q = self._queue_lengths[phase]
        w = self._wait_times[phase]
        state = self._rl.make_state(queue_length=q, waiting_time=w, phase=phase)
        action = self._rl.best_action(state)
        green_time = self._rl.get_green_time(action)
        return {
            "green_time": green_time,
            "action": action,
            "state": state.to_tuple(),
            "q_values": self._rl._get_q(state.to_tuple()).tolist(),
        }

    # ── Combined Decision ────────────────────────────────────────────────────

    def decide(self) -> dict:
        """
        Main decision function. Returns green time recommendation based on mode.
        Also performs RL learning step if in RL/auto mode.
        """
        phase = self._state.active_phase
        q = sum(self._queue_lengths)
        w = max(self._wait_times)

        if self._mode == "manual":
            green_time = 30.0
            method = "manual"
        elif self._mode == "fuzzy":
            green_time = self._fuzzy.decide(queue_length=q, waiting_time=w)
            method = "fuzzy"
        elif self._mode == "rl":
            state = self._rl.make_state(q, w, phase)
            action = self._rl.select_action(state)
            green_time = float(self._rl.get_green_time(action))
            self._do_rl_learn(state.to_tuple(), action, q, w)
            method = "rl"
        else:  # auto
            state = self._rl.make_state(q, w, phase)
            action = self._rl.select_action(state)
            rl_green = float(self._rl.get_green_time(action))
            fuzzy_green = self._fuzzy.decide(queue_length=q, waiting_time=w)
            # Blend: 70% RL + 30% Fuzzy (safety net)
            green_time = 0.7 * rl_green + 0.3 * fuzzy_green
            self._do_rl_learn(state.to_tuple(), action, q, w)
            method = "auto"

        green_time = round(max(10, min(70, green_time)), 1)

        # Update state
        self._state.phases[phase].green_time = green_time
        self._state.phases[phase].color = "green"
        self._state.phases[phase].queue_length = self._queue_lengths[phase]
        self._state.phases[phase].avg_wait = round(self._wait_times[phase], 1)
        self._state.phases[1 - phase].color = "red"
        self._state.phases[1 - phase].queue_length = self._queue_lengths[1 - phase]
        self._state.phases[1 - phase].avg_wait = round(self._wait_times[1 - phase], 1)

        return {
            "green_time": green_time,
            "method": method,
            "phase": phase,
            "queue": q,
            "wait": round(w, 1),
        }

    def advance_phase(self) -> None:
        """Switch to next phase."""
        self._state.active_phase = 1 - self._state.active_phase
        self._state.cycle_count += 1

    def _do_rl_learn(self, state: tuple, action: int, q: float, w: float) -> None:
        """Perform one Q-learning update from the previous step."""
        if self._prev_rl_state is not None and self._prev_rl_action is not None:
            reward = RLAgent.compute_reward(
                avg_wait=w,
                total_queue=q,
                did_switch=True,
                throughput=sum(self._throughput),
            )
            exp = RLExperience(
                state=self._prev_rl_state,
                action=self._prev_rl_action,
                reward=reward,
                next_state=state,
            )
            self._rl.update(exp)
            self._episode_reward += reward

        self._prev_rl_state = state
        self._prev_rl_action = action

    # ── Simulation ───────────────────────────────────────────────────────────

    def run_simulation(
        self,
        mode: str = "fuzzy",
        episodes: int = 10,
        cycles: int = 100,
        arrival_rates: tuple[float, float] = (0.3, 0.25),
        fixed_green: float = 30.0,
    ) -> SimulationResult:
        """Run simulation episodes and return results."""
        sim = TrafficSimulator(arrival_rates=arrival_rates)
        total_reward = 0.0
        total_wait = 0.0
        total_cleared = 0
        total_queue = 0.0

        for ep in range(episodes):
            if mode == "fuzzy":
                def policy(obs):
                    return self._fuzzy.decide(obs["total_queue"], obs["avg_wait"])
            elif mode == "rl":
                def policy(obs):
                    s = self._rl.make_state(obs["total_queue"], obs["avg_wait"], obs["phase"])
                    a = self._rl.select_action(s)
                    return float(self._rl.get_green_time(a))
            else:  # fixed
                def policy(obs):
                    return fixed_green

            result = sim.run_episode(policy, max_cycles=cycles, seed=ep)
            total_reward += result["avg_reward"]
            total_wait += result["avg_wait"]
            total_cleared += result["total_cleared"]
            total_queue += result["final_queue"]

        n = max(episodes, 1)
        return SimulationResult(
            mode=mode,
            episodes=episodes,
            avg_reward=round(total_reward / n, 2),
            avg_wait=round(total_wait / n, 2),
            total_cleared=total_cleared,
            avg_queue=round(total_queue / n, 1),
        )

    # ── RL Training ──────────────────────────────────────────────────────────

    def train_rl(self, episodes: int = 200, cycles: int = 100) -> dict:
        """Train RL agent on simulator."""
        sim = TrafficSimulator()

        for ep in range(episodes):
            sim.reset(seed=ep)
            ep_reward = 0.0
            prev_state = None
            prev_action = None

            for _ in range(cycles):
                obs = sim.get_obs()
                state = self._rl.make_state(obs["total_queue"], obs["avg_wait"], obs["phase"])

                action = self._rl.select_action(state)
                green_time = float(self._rl.get_green_time(action))
                info = sim.step(green_time, switch_phase=True)

                reward = RLAgent.compute_reward(
                    avg_wait=info["avg_wait"],
                    total_queue=info["total_queue"],
                    did_switch=info["did_switch"],
                    throughput=info["cleared"],
                )

                if prev_state is not None:
                    exp = RLExperience(
                        state=prev_state,
                        action=prev_action,
                        reward=reward,
                        next_state=state.to_tuple(),
                    )
                    self._rl.update(exp)

                prev_state = state.to_tuple()
                prev_action = action
                ep_reward += reward

            self._rl.end_episode(ep_reward)

        self._save_state()
        return self._rl.get_q_table_stats()

    # ── GA Optimization ──────────────────────────────────────────────────────

    def start_ga(
        self,
        population_size: int = 50,
        generations: int = 30,
        eval_episodes: int = 5,
        eval_cycles: int = 50,
    ) -> None:
        """Start GA optimization in background thread."""
        if self._ga.running:
            logger.warning("GA already running")
            return

        config = GAConfig(
            population_size=population_size,
            generations=generations,
            eval_episodes=eval_episodes,
            eval_cycles=eval_cycles,
        )
        self._ga = GeneticOptimizer(config)

        def _run():
            logger.info("GA: starting optimization (%d gens, pop=%d)",
                        generations, population_size)
            best = self._ga.run(progress_callback=self._ga_progress)
            # Apply best chromosome to fuzzy controller
            self._fuzzy.from_chromosome(best.genes)
            self._save_state()
            logger.info("GA: done, best fitness=%.2f", best.fitness)

        self._ga_thread = threading.Thread(target=_run, daemon=True)
        self._ga_thread.start()

    def stop_ga(self) -> None:
        self._ga.stop()

    def _ga_progress(self, gen: int, total: int, fitness: float) -> None:
        logger.info("GA: gen %d/%d, best=%.2f", gen, total, fitness)

    def get_ga_status(self) -> GAStatus:
        return GAStatus(
            running=self._ga.running,
            generation=self._ga.generation,
            total_generations=self._ga.config.generations if self._ga else 0,
            best_fitness=self._ga.best.fitness if self._ga.best else 0.0,
            avg_fitness=self._ga.history[-1]["avg_fitness"] if self._ga.history else 0.0,
            history=self._ga.history[-50:],  # last 50 generations
        )

    # ── State ────────────────────────────────────────────────────────────────

    def get_state(self) -> TrafficLightState:
        self._state.mode = self._mode
        return self._state

    def get_rl_status(self) -> RLStatus:
        stats = self._rl.get_q_table_stats()
        return RLStatus(**{k: stats[k] for k in RLStatus.model_fields if k in stats})

    def get_fuzzy_params(self) -> dict:
        return self._fuzzy.get_params()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        try:
            self._rl.save(RL_SAVE_PATH)
            params = self._fuzzy.get_params()
            FUZZY_SAVE_PATH.write_text(json.dumps(params, indent=2))
            logger.info("TrafficLight: state saved")
        except Exception as e:
            logger.warning("TrafficLight: save failed: %s", e)

    def _load_state(self) -> None:
        try:
            self._rl.load(RL_SAVE_PATH)
            if FUZZY_SAVE_PATH.exists():
                params = json.loads(FUZZY_SAVE_PATH.read_text())
                self._fuzzy.set_params(params)
            logger.info("TrafficLight: state loaded")
        except Exception as e:
            logger.warning("TrafficLight: load failed (using defaults): %s", e)


# Singleton
traffic_light_service = TrafficLightService()
