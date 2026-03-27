"""
Traffic Intersection Simulator.

Simulates a 2-phase intersection with stochastic vehicle arrivals.
Used for:
  1. Training the RL agent (online learning)
  2. Evaluating GA fitness (offline optimization)

Model:
  - 2 approaches (North-South = phase 0, East-West = phase 1)
  - Vehicles arrive following Poisson process
  - Green phase clears vehicles at a fixed saturation rate
  - Vehicles in red queue accumulate wait time
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LaneState:
    queue: int = 0                  # vehicles waiting
    total_wait: float = 0.0         # cumulative wait (seconds) of all vehicles
    cleared: int = 0                # vehicles that passed through
    avg_wait: float = 0.0           # average wait of cleared vehicles


@dataclass
class SimState:
    """Full intersection state at a point in time."""
    lanes: list[LaneState] = field(default_factory=lambda: [LaneState(), LaneState()])
    current_phase: int = 0          # which lane has green (0 or 1)
    time: float = 0.0               # simulation clock (seconds)
    cycle_count: int = 0            # number of completed signal cycles


class TrafficSimulator:
    """
    Simulates a 2-phase signalized intersection.

    Parameters:
        arrival_rates: vehicles/second for each phase [phase0, phase1]
        saturation_rate: vehicles/second that can clear during green
        yellow_time: seconds of yellow (lost time) between phases
        min_green: minimum green time
        max_green: maximum green time
    """

    def __init__(
        self,
        arrival_rates: tuple[float, float] = (0.3, 0.25),
        saturation_rate: float = 0.5,
        yellow_time: float = 3.0,
        min_green: float = 10.0,
        max_green: float = 70.0,
    ) -> None:
        self.arrival_rates = list(arrival_rates)
        self.saturation_rate = saturation_rate
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green

        self._state = SimState()
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> SimState:
        """Reset simulation to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = SimState()
        # Random initial queues
        for i in range(2):
            self._state.lanes[i].queue = self._rng.integers(0, 10)
        return self._state

    @property
    def state(self) -> SimState:
        return self._state

    def get_obs(self) -> dict:
        """Observable state for the RL agent."""
        s = self._state
        green_lane = s.lanes[s.current_phase]
        red_lane = s.lanes[1 - s.current_phase]
        return {
            "queue_green": green_lane.queue,
            "queue_red": red_lane.queue,
            "total_queue": green_lane.queue + red_lane.queue,
            "wait_green": green_lane.avg_wait,
            "wait_red": red_lane.avg_wait,
            "avg_wait": (green_lane.avg_wait + red_lane.avg_wait) / 2,
            "phase": s.current_phase,
            "time": s.time,
            "cycle": s.cycle_count,
        }

    def step(self, green_time: float, switch_phase: bool = True) -> dict:
        """
        Execute one signal cycle:
          1. Green phase runs for `green_time` seconds
          2. Yellow phase (lost time)
          3. Optionally switch to next phase

        Returns step info dict with reward components.
        """
        green_time = float(np.clip(green_time, self.min_green, self.max_green))
        s = self._state
        green_idx = s.current_phase
        red_idx = 1 - green_idx

        # ── During green phase ──
        duration = green_time + self.yellow_time

        # Arrivals during the cycle (Poisson)
        arrivals_green = self._rng.poisson(self.arrival_rates[green_idx] * duration)
        arrivals_red = self._rng.poisson(self.arrival_rates[red_idx] * duration)

        # Vehicles cleared during green (up to queue + arrivals)
        max_clearable = int(self.saturation_rate * green_time)
        available = s.lanes[green_idx].queue + arrivals_green
        cleared = min(max_clearable, available)

        # Update green lane
        s.lanes[green_idx].queue = available - cleared
        s.lanes[green_idx].cleared += cleared
        # Average wait for cleared vehicles (simplified: half of green_time)
        if cleared > 0:
            wait_estimate = green_time / 2
            prev_total = s.lanes[green_idx].avg_wait * max(s.lanes[green_idx].cleared - cleared, 0)
            s.lanes[green_idx].avg_wait = (prev_total + cleared * wait_estimate) / max(s.lanes[green_idx].cleared, 1)

        # Update red lane (vehicles accumulate)
        s.lanes[red_idx].queue += arrivals_red
        # Wait time accumulates for all vehicles in red queue
        s.lanes[red_idx].total_wait += s.lanes[red_idx].queue * duration
        if s.lanes[red_idx].queue > 0:
            s.lanes[red_idx].avg_wait = s.lanes[red_idx].total_wait / max(s.lanes[red_idx].queue, 1)

        # Advance time
        s.time += duration
        s.cycle_count += 1

        # Switch phase
        did_switch = False
        if switch_phase:
            s.current_phase = red_idx
            did_switch = True

        # Compute reward components
        total_queue = s.lanes[0].queue + s.lanes[1].queue
        avg_wait = (s.lanes[0].avg_wait + s.lanes[1].avg_wait) / 2

        info = {
            "cleared": cleared,
            "arrivals_green": arrivals_green,
            "arrivals_red": arrivals_red,
            "total_queue": total_queue,
            "avg_wait": avg_wait,
            "did_switch": did_switch,
            "green_time_used": green_time,
            "duration": duration,
        }

        return info

    def run_episode(
        self,
        policy_fn,
        max_cycles: int = 100,
        seed: int | None = None,
    ) -> dict:
        """
        Run a full episode using a policy function.

        policy_fn(obs) → green_time (float, seconds)

        Returns episode summary.
        """
        self.reset(seed)
        total_reward = 0.0
        total_cleared = 0
        total_wait_sum = 0.0

        for _ in range(max_cycles):
            obs = self.get_obs()
            green_time = policy_fn(obs)
            info = self.step(green_time, switch_phase=True)

            from .rl_agent import RLAgent
            reward = RLAgent.compute_reward(
                avg_wait=info["avg_wait"],
                total_queue=info["total_queue"],
                did_switch=info["did_switch"],
                throughput=info["cleared"],
            )
            total_reward += reward
            total_cleared += info["cleared"]
            total_wait_sum += info["avg_wait"]

        return {
            "total_reward": total_reward,
            "avg_reward": total_reward / max_cycles,
            "total_cleared": total_cleared,
            "avg_wait": total_wait_sum / max_cycles,
            "cycles": max_cycles,
            "final_queue": sum(l.queue for l in self._state.lanes),
        }

    def set_arrival_rates(self, rates: tuple[float, float]) -> None:
        self.arrival_rates = list(rates)
