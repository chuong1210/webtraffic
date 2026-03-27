"""
Pydantic models for Traffic Light Optimization API.
"""

from __future__ import annotations
from pydantic import BaseModel, Field


class TrafficLightPhase(BaseModel):
    """Current state of one traffic light phase."""
    phase_id: int = 0
    color: str = "red"                  # "red" | "yellow" | "green"
    green_time: float = 30.0            # recommended green duration (s)
    remaining: float = 0.0              # seconds remaining in current phase
    queue_length: int = 0               # vehicles waiting
    avg_wait: float = 0.0               # average wait time (s)


class TrafficLightState(BaseModel):
    """Full traffic light intersection state."""
    phases: list[TrafficLightPhase] = Field(default_factory=lambda: [
        TrafficLightPhase(phase_id=0, color="red"),
        TrafficLightPhase(phase_id=1, color="red"),
    ])
    active_phase: int = 0
    cycle_count: int = 0
    mode: str = "manual"                # "manual" | "fuzzy" | "rl" | "auto"
    time_elapsed: float = 0.0


class FuzzyDecision(BaseModel):
    """Result of fuzzy inference."""
    green_time: float
    queue_input: float
    wait_input: float
    firing_rules: list[str] = []


class RLStatus(BaseModel):
    """RL agent training status."""
    states_visited: int = 0
    total_steps: int = 0
    total_episodes: int = 0
    epsilon: float = 0.15
    avg_reward: float = 0.0
    avg_q: float = 0.0
    max_q: float = 0.0


class GAStatus(BaseModel):
    """GA optimization status."""
    running: bool = False
    generation: int = 0
    total_generations: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    history: list[dict] = []


class SimulationRequest(BaseModel):
    """Request to run a traffic simulation."""
    mode: str = "fuzzy"                 # "fuzzy" | "rl" | "fixed"
    episodes: int = 10
    cycles_per_episode: int = 100
    arrival_rate_0: float = 0.3
    arrival_rate_1: float = 0.25
    fixed_green: float = 30.0           # used when mode="fixed"


class SimulationResult(BaseModel):
    """Result of simulation run."""
    mode: str
    episodes: int
    avg_reward: float
    avg_wait: float
    total_cleared: int
    avg_queue: float


class GARequest(BaseModel):
    """Request to start GA optimization."""
    population_size: int = 50
    generations: int = 30
    eval_episodes: int = 5
    eval_cycles: int = 50


class TrafficLightConfig(BaseModel):
    """Configuration update for traffic light system."""
    mode: str | None = None             # "manual" | "fuzzy" | "rl" | "auto"
    arrival_rate_0: float | None = None
    arrival_rate_1: float | None = None
    min_green: float | None = None
    max_green: float | None = None
    rl_alpha: float | None = None
    rl_gamma: float | None = None
    rl_epsilon: float | None = None
