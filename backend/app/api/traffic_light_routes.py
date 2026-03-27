"""
Traffic Light Optimization API routes.

Endpoints:
  GET  /state          - current traffic light state
  POST /decide         - get green time recommendation
  POST /advance        - switch to next phase
  POST /mode           - change operating mode
  GET  /fuzzy          - fuzzy controller params & decision
  GET  /rl/status      - RL agent stats
  POST /rl/train       - train RL on simulator
  POST /ga/start       - start GA optimization
  POST /ga/stop        - stop GA
  GET  /ga/status      - GA progress
  POST /simulate       - run comparison simulation
"""

from __future__ import annotations
from fastapi import APIRouter

from app.models.traffic_light_model import (
    TrafficLightState, TrafficLightConfig, FuzzyDecision,
    RLStatus, GAStatus, GARequest, SimulationRequest, SimulationResult,
)
from app.models.detection_model import SuccessResponse
from app.services.traffic_light_service import traffic_light_service as tls

router = APIRouter(prefix="/api/v1/traffic-light", tags=["traffic-light"])


@router.get("/state", response_model=TrafficLightState)
async def get_state():
    return tls.get_state()


@router.post("/decide")
async def decide():
    """Get green time recommendation from current mode."""
    return tls.decide()


@router.post("/advance")
async def advance_phase():
    """Switch to next traffic light phase."""
    tls.advance_phase()
    return SuccessResponse(success=True, message="Phase advanced")


@router.post("/mode")
async def set_mode(body: TrafficLightConfig):
    if body.mode:
        tls.set_mode(body.mode)
    return SuccessResponse(success=True, message=f"Mode: {body.mode}")


# ── Fuzzy ────────────────────────────────────────────────────────────────────

@router.get("/fuzzy/params")
async def get_fuzzy_params():
    return tls.get_fuzzy_params()


@router.get("/fuzzy/decide")
async def fuzzy_decide(queue: float = 10, wait: float = 30):
    """Interactive fuzzy decision with custom inputs."""
    from app.ml.traffic_light.fuzzy_controller import FuzzyController
    ctrl = FuzzyController()
    ctrl.set_params(tls.get_fuzzy_params())
    green = ctrl.decide(queue_length=queue, waiting_time=wait)
    return FuzzyDecision(green_time=round(green, 1), queue_input=queue, wait_input=wait)


# ── RL ───────────────────────────────────────────────────────────────────────

@router.get("/rl/status", response_model=RLStatus)
async def rl_status():
    return tls.get_rl_status()


@router.post("/rl/train")
async def rl_train(episodes: int = 200, cycles: int = 100):
    """Train RL agent on traffic simulator."""
    result = tls.train_rl(episodes=episodes, cycles=cycles)
    return {"success": True, "stats": result}


# ── GA ───────────────────────────────────────────────────────────────────────

@router.get("/ga/status", response_model=GAStatus)
async def ga_status():
    return tls.get_ga_status()


@router.post("/ga/start")
async def ga_start(body: GARequest):
    tls.start_ga(
        population_size=body.population_size,
        generations=body.generations,
        eval_episodes=body.eval_episodes,
        eval_cycles=body.eval_cycles,
    )
    return SuccessResponse(success=True, message="GA optimization started")


@router.post("/ga/stop")
async def ga_stop():
    tls.stop_ga()
    return SuccessResponse(success=True, message="GA stopped")


# ── Simulation ───────────────────────────────────────────────────────────────

@router.post("/simulate", response_model=list[SimulationResult])
async def simulate(body: SimulationRequest):
    """
    Run simulation and compare modes.
    Returns results for: fixed, fuzzy, rl (if trained).
    """
    rates = (body.arrival_rate_0, body.arrival_rate_1)
    results = []

    # Fixed baseline
    r_fixed = tls.run_simulation(
        mode="fixed", episodes=body.episodes, cycles=body.cycles_per_episode,
        arrival_rates=rates, fixed_green=body.fixed_green,
    )
    results.append(r_fixed)

    # Fuzzy
    r_fuzzy = tls.run_simulation(
        mode="fuzzy", episodes=body.episodes, cycles=body.cycles_per_episode,
        arrival_rates=rates,
    )
    results.append(r_fuzzy)

    # RL (if trained)
    rl_stats = tls.get_rl_status()
    if rl_stats.total_episodes > 0:
        r_rl = tls.run_simulation(
            mode="rl", episodes=body.episodes, cycles=body.cycles_per_episode,
            arrival_rates=rates,
        )
        results.append(r_rl)

    return results
