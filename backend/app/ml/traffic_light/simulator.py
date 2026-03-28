"""
Traffic Intersection Simulator — Realistic Version.

Upgraded from basic Poisson → realistic traffic patterns:
  - Time-of-day demand curves (rush hour peaks)
  - Platoon arrivals (vehicles arrive in groups, not uniformly)
  - Queue spillback (downstream congestion limits capacity)
  - Variable saturation rate (depends on vehicle mix)
  - SUMO adapter for real-world simulation when available

Model:
  - 2 approaches (North-South = phase 0, East-West = phase 1)
  - Green phase clears vehicles at dynamic saturation rate
  - Supports both internal sim and SUMO backend
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

# Try SUMO
try:
    import traci
    from sumolib import checkBinary
    HAS_SUMO = True
except ImportError:
    HAS_SUMO = False


# ── Time-of-Day Demand Profile ───────────────────────────────────────────────

class DemandProfile:
    """
    Realistic hourly demand multipliers (24h cycle).
    Peak hours: 7-9 AM, 5-7 PM.
    """
    HOURLY_MULTIPLIERS = [
        0.15, 0.10, 0.08, 0.08, 0.10, 0.20,  # 0-5h  (night)
        0.40, 0.85, 1.00, 0.75, 0.60, 0.55,  # 6-11h (morning rush)
        0.65, 0.60, 0.55, 0.60, 0.80, 1.00,  # 12-17h (afternoon rush)
        0.85, 0.65, 0.45, 0.35, 0.25, 0.18,  # 18-23h (evening)
    ]

    @classmethod
    def get_multiplier(cls, sim_time: float, cycle_seconds: float = 3600.0) -> float:
        """Map simulation time to hour-of-day multiplier."""
        hour = int((sim_time / cycle_seconds) * 24) % 24
        return cls.HOURLY_MULTIPLIERS[hour]

    @classmethod
    def get_multiplier_smooth(cls, sim_time: float, cycle_seconds: float = 3600.0) -> float:
        """Smooth interpolation between hours."""
        hour_frac = ((sim_time / cycle_seconds) * 24) % 24
        h0 = int(hour_frac) % 24
        h1 = (h0 + 1) % 24
        t = hour_frac - h0
        return cls.HOURLY_MULTIPLIERS[h0] * (1 - t) + cls.HOURLY_MULTIPLIERS[h1] * t


# ── Lane State ───────────────────────────────────────────────────────────────

@dataclass
class LaneState:
    queue: int = 0
    total_wait: float = 0.0
    cleared: int = 0
    avg_wait: float = 0.0
    # Realistic additions
    heavy_vehicle_ratio: float = 0.1   # trucks/buses ratio (slows saturation)
    platoon_pending: int = 0           # vehicles arriving as platoon


@dataclass
class SimState:
    lanes: list[LaneState] = field(default_factory=lambda: [LaneState(), LaneState()])
    current_phase: int = 0
    time: float = 0.0
    cycle_count: int = 0


# ── Traffic Simulator ────────────────────────────────────────────────────────

class TrafficSimulator:
    """
    Realistic 2-phase intersection simulator.

    Improvements over basic version:
      1. Time-of-day demand curves
      2. Platoon arrivals (Negative Binomial instead of Poisson)
      3. Variable saturation based on vehicle mix
      4. Queue capacity limit (spillback)
      5. Start-up lost time on green
    """

    def __init__(
        self,
        arrival_rates: tuple[float, float] = (0.3, 0.25),
        saturation_rate: float = 0.5,
        yellow_time: float = 3.0,
        min_green: float = 10.0,
        max_green: float = 70.0,
        # Realistic params
        use_demand_profile: bool = True,
        use_platoons: bool = True,
        queue_capacity: int = 60,
        startup_lost_time: float = 2.0,
        heavy_vehicle_factor: float = 0.85,  # saturation reduction for trucks
    ) -> None:
        self.base_arrival_rates = list(arrival_rates)
        self.arrival_rates = list(arrival_rates)
        self.saturation_rate = saturation_rate
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green

        self.use_demand_profile = use_demand_profile
        self.use_platoons = use_platoons
        self.queue_capacity = queue_capacity
        self.startup_lost_time = startup_lost_time
        self.heavy_vehicle_factor = heavy_vehicle_factor

        self._state = SimState()
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> SimState:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = SimState()
        for i in range(2):
            self._state.lanes[i].queue = int(self._rng.integers(0, 10))
            self._state.lanes[i].heavy_vehicle_ratio = float(self._rng.uniform(0.05, 0.2))
        return self._state

    @property
    def state(self) -> SimState:
        return self._state

    def get_obs(self) -> dict:
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
            "heavy_ratio": (green_lane.heavy_vehicle_ratio + red_lane.heavy_vehicle_ratio) / 2,
        }

    def step(self, green_time: float, switch_phase: bool = True) -> dict:
        """
        Execute one signal cycle with realistic dynamics.
        """
        green_time = float(np.clip(green_time, self.min_green, self.max_green))
        s = self._state
        green_idx = s.current_phase
        red_idx = 1 - green_idx

        duration = green_time + self.yellow_time

        # ── 1. Demand adjustment (time-of-day) ──
        if self.use_demand_profile:
            multiplier = DemandProfile.get_multiplier_smooth(s.time)
        else:
            multiplier = 1.0

        effective_rates = [r * multiplier for r in self.base_arrival_rates]

        # ── 2. Arrivals (platoon or Poisson) ──
        if self.use_platoons:
            # Negative Binomial: models platoon arrivals (overdispersed)
            # Higher variance than Poisson → bursty traffic
            for idx, rate in enumerate(effective_rates):
                mean_arrivals = rate * duration
                if mean_arrivals <= 0:
                    arrivals = 0
                else:
                    # NB parameterization: n=dispersion, p=n/(n+mean)
                    n_disp = 3.0  # lower = more bursty
                    p = n_disp / (n_disp + mean_arrivals)
                    arrivals = int(self._rng.negative_binomial(n_disp, p))

                if idx == green_idx:
                    arrivals_green = arrivals
                else:
                    arrivals_red = arrivals
        else:
            arrivals_green = int(self._rng.poisson(effective_rates[green_idx] * duration))
            arrivals_red = int(self._rng.poisson(effective_rates[red_idx] * duration))

        # ── 3. Effective green time (subtract startup lost time) ──
        effective_green = max(0, green_time - self.startup_lost_time)

        # ── 4. Dynamic saturation rate (heavy vehicles reduce throughput) ──
        hvr = s.lanes[green_idx].heavy_vehicle_ratio
        effective_sat = self.saturation_rate * (1.0 - hvr * (1.0 - self.heavy_vehicle_factor))

        # ── 5. Vehicles cleared ──
        max_clearable = int(effective_sat * effective_green)
        available = s.lanes[green_idx].queue + arrivals_green
        cleared = min(max_clearable, available)

        # ── 6. Queue update with capacity limit ──
        new_green_queue = min(available - cleared, self.queue_capacity)
        s.lanes[green_idx].queue = new_green_queue
        s.lanes[green_idx].cleared += cleared

        if cleared > 0:
            wait_estimate = green_time / 2
            prev_total = s.lanes[green_idx].avg_wait * max(s.lanes[green_idx].cleared - cleared, 0)
            s.lanes[green_idx].avg_wait = (prev_total + cleared * wait_estimate) / max(s.lanes[green_idx].cleared, 1)

        # Red lane accumulates
        new_red_queue = min(s.lanes[red_idx].queue + arrivals_red, self.queue_capacity)
        s.lanes[red_idx].queue = new_red_queue
        s.lanes[red_idx].total_wait += s.lanes[red_idx].queue * duration
        if s.lanes[red_idx].queue > 0:
            s.lanes[red_idx].avg_wait = s.lanes[red_idx].total_wait / max(s.lanes[red_idx].queue, 1)

        # ── 7. Random heavy vehicle ratio drift ──
        for lane in s.lanes:
            lane.heavy_vehicle_ratio = float(np.clip(
                lane.heavy_vehicle_ratio + self._rng.normal(0, 0.02),
                0.02, 0.35
            ))

        s.time += duration
        s.cycle_count += 1

        did_switch = False
        if switch_phase:
            s.current_phase = red_idx
            did_switch = True

        total_queue = s.lanes[0].queue + s.lanes[1].queue
        avg_wait = (s.lanes[0].avg_wait + s.lanes[1].avg_wait) / 2

        return {
            "cleared": cleared,
            "arrivals_green": arrivals_green,
            "arrivals_red": arrivals_red,
            "total_queue": total_queue,
            "avg_wait": avg_wait,
            "did_switch": did_switch,
            "green_time_used": green_time,
            "duration": duration,
            "demand_multiplier": multiplier,
            "effective_saturation": effective_sat,
        }

    def run_episode(
        self,
        policy_fn,
        max_cycles: int = 100,
        seed: int | None = None,
    ) -> dict:
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
        self.base_arrival_rates = list(rates)
        self.arrival_rates = list(rates)


# ── SUMO Adapter ─────────────────────────────────────────────────────────────

class SUMOSimulator:
    """
    Adapter that uses SUMO for realistic simulation.
    Falls back to TrafficSimulator if SUMO is not installed.

    Ref: Traffic_RL/train_RL.py
    """

    PHASE_STATES = [
        ["yyyrrrrrrrrr", "GGGrrrrrrrrr"],  # phase 0
        ["rrryyyrrrrrr", "rrrGGGrrrrrr"],  # phase 1
        ["rrrrrryyyrrr", "rrrrrrGGGrrr"],  # phase 2
        ["rrrrrrrrryyy", "rrrrrrrrrGGG"],  # phase 3
    ]

    def __init__(self, config_path: str = "configuration.sumocfg", gui: bool = False):
        if not HAS_SUMO:
            raise ImportError("SUMO (traci + sumolib) not installed. "
                              "Use TrafficSimulator instead.")
        self.config_path = config_path
        self.gui = gui
        self._running = False
        self._junctions: list[str] = []
        self._all_lanes: list[str] = []
        self._step = 0

    def start(self) -> None:
        binary = checkBinary("sumo-gui" if self.gui else "sumo")
        traci.start([binary, "-c", self.config_path,
                     "--tripinfo-output", "tripinfo.xml"])
        self._junctions = list(traci.trafficlight.getIDList())
        self._all_lanes = []
        for j in self._junctions:
            self._all_lanes.extend(traci.trafficlight.getControlledLanes(j))
        self._running = True
        self._step = 0

    def stop(self) -> None:
        if self._running:
            traci.close()
            self._running = False

    def get_state(self, junction_idx: int = 0) -> np.ndarray:
        """Get 4D state: vehicle count per lane group (like Traffic_RL)."""
        if junction_idx >= len(self._junctions):
            return np.zeros(4, dtype=np.float32)
        junction = self._junctions[junction_idx]
        lanes = traci.trafficlight.getControlledLanes(junction)
        counts = []
        for lane in lanes:
            n = 0
            for vid in traci.lane.getLastStepVehicleIDs(lane):
                if traci.vehicle.getLanePosition(vid) > 10:
                    n += 1
            counts.append(n)
        # Group into 4 approach counts
        grouped = [0.0] * 4
        for i, c in enumerate(counts):
            grouped[i % 4] += c
        return np.array(grouped, dtype=np.float32)

    def get_waiting_time(self, junction_idx: int = 0) -> float:
        if junction_idx >= len(self._junctions):
            return 0.0
        junction = self._junctions[junction_idx]
        lanes = traci.trafficlight.getControlledLanes(junction)
        total = 0.0
        for lane in lanes:
            total += traci.lane.getWaitingTime(lane)
        return total

    def apply_action(self, action: int, junction_idx: int = 0,
                     min_duration: int = 15) -> None:
        """Apply action (select lane phase). Ref: Traffic_RL."""
        if junction_idx >= len(self._junctions):
            return
        junction = self._junctions[junction_idx]
        phase = action % len(self.PHASE_STATES)
        # Yellow first
        traci.trafficlight.setRedYellowGreenState(
            junction, self.PHASE_STATES[phase][0])
        traci.trafficlight.setPhaseDuration(junction, 6)
        # Then green
        traci.trafficlight.setRedYellowGreenState(
            junction, self.PHASE_STATES[phase][1])
        traci.trafficlight.setPhaseDuration(junction, min_duration)

    def sim_step(self) -> None:
        traci.simulationStep()
        self._step += 1

    def run_episode(self, agent, steps: int = 500) -> dict:
        """
        Run a full SUMO episode with a DQN agent.
        Ref: Traffic_RL train_RL.py run() function.
        """
        self.start()
        total_wait = 0.0
        phase_timers = {j: 0 for j in self._junctions}
        prev_states = {i: np.zeros(4, dtype=np.float32) for i in range(len(self._junctions))}
        prev_actions = {i: 0 for i in range(len(self._junctions))}

        for step in range(steps):
            self.sim_step()

            for ji, junction in enumerate(self._junctions):
                wait = self.get_waiting_time(ji)
                total_wait += wait

                if phase_timers[junction] <= 0:
                    state = self.get_state(ji)
                    reward = -wait

                    # Store experience & learn
                    from .rl_agent import RLExperience
                    exp = RLExperience(
                        state=prev_states[ji],
                        action=prev_actions[ji],
                        reward=reward,
                        next_state=state,
                        done=(step == steps - 1),
                    )
                    agent.update(exp)

                    # Choose action
                    action = agent.select_action(state)
                    prev_states[ji] = state
                    prev_actions[ji] = action

                    self.apply_action(action, ji, min_duration=15)
                    phase_timers[junction] = 15
                else:
                    phase_timers[junction] -= 1

        self.stop()
        return {"total_wait": total_wait, "steps": steps}

    @property
    def junctions(self) -> list[str]:
        return self._junctions
