"""
Ambulance scenario manager for Surakshit Bharat Q-JEPA.
Creates reproducible emergency-routing test scenarios.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np

from src.simulation.road_network import RoadNetwork
from src.simulation.traffic_simulator import ScenarioType, TrafficSimulator
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AmbulanceScenario:
    """An ambulance routing scenario definition."""
    scenario_id: str
    scenario_type: ScenarioType
    origin: str
    destination: str
    description: str
    seed: int = 42
    expected_min_travel_s: Optional[float] = None


@dataclass
class AmbulanceJourney:
    """Record of an ambulance journey."""
    scenario: AmbulanceScenario
    route: List[str]
    total_travel_time_s: float
    arrival_step: int
    reroute_count: int = 0
    steps_taken: List[Dict] = field(default_factory=list)
    completed: bool = False


def create_scenario_suite(network: RoadNetwork, seed: int = 42) -> List[AmbulanceScenario]:
    """Create the seven required scenario types."""
    rng = np.random.default_rng(seed)
    nodes = network.node_list()
    n = len(nodes)

    def pick_pair() -> Tuple[str, str]:
        i, j = rng.choice(n, size=2, replace=False)
        return nodes[int(i)], nodes[int(j)]

    scenarios = []
    types = [
        (ScenarioType.NORMAL, "Normal city traffic"),
        (ScenarioType.RUSH_HOUR, "Rush-hour congestion"),
        (ScenarioType.ROAD_BLOCKAGE, "Sudden road blockage"),
        (ScenarioType.QUEUE_SPILLBACK, "Queue spillback across intersections"),
        (ScenarioType.MISSING_OBS, "Missing / delayed observations"),
        (ScenarioType.NOISY_OBS, "Noisy traffic measurements"),
        (ScenarioType.DYNAMIC_REROUTE, "Route becomes slower mid-journey"),
    ]
    for idx, (stype, desc) in enumerate(types):
        origin, dest = pick_pair()
        scenarios.append(
            AmbulanceScenario(
                scenario_id=f"S{idx+1:02d}_{stype.value}",
                scenario_type=stype,
                origin=origin,
                destination=dest,
                description=desc,
                seed=seed + idx,
            )
        )

    return scenarios
