"""
Incident generator for Surakshit Bharat Q-JEPA.
Generates random and scheduled incidents on a road network.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.simulation.road_network import RoadNetwork
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Incident:
    incident_id: str
    edge_id: str
    u: str
    v: str
    start_step: int
    duration_steps: int
    incident_type: str  # "collision", "breakdown", "construction", "closure"
    speed_factor: float  # multiplier on baseline speed


class IncidentGenerator:
    """Generates scheduled and random incidents."""

    def __init__(self, network: RoadNetwork, seed: int = 42,
                 incident_prob: float = 0.02, closure_prob: float = 0.005):
        self.network = network
        self.seed = seed
        self.incident_prob = incident_prob
        self.closure_prob = closure_prob
        self._rng = np.random.default_rng(seed)
        self.active_incidents: List[Incident] = []
        self._counter = 0

    def step(self, current_step: int) -> List[Incident]:
        """Check for new incidents and expire old ones. Returns new incidents."""
        edges = self.network.edge_list()
        new_incidents = []

        for u, v in edges:
            if float(self._rng.random()) < self.incident_prob:
                inc = Incident(
                    incident_id=f"INC{self._counter:04d}",
                    edge_id=f"{u}->{v}",
                    u=u, v=v,
                    start_step=current_step,
                    duration_steps=int(self._rng.integers(3, 20)),
                    incident_type=str(self._rng.choice(["collision", "breakdown", "debris"])),
                    speed_factor=float(self._rng.uniform(0.2, 0.5)),
                )
                self._counter += 1
                self.active_incidents.append(inc)
                new_incidents.append(inc)
                self.network.update_edge_state(u, v, has_incident=True, step=current_step)
                logger.debug("New incident %s on %s->%s", inc.incident_id, u, v)

            if float(self._rng.random()) < self.closure_prob:
                inc = Incident(
                    incident_id=f"INC{self._counter:04d}",
                    edge_id=f"{u}->{v}",
                    u=u, v=v,
                    start_step=current_step,
                    duration_steps=int(self._rng.integers(10, 50)),
                    incident_type="closure",
                    speed_factor=0.0,
                )
                self._counter += 1
                self.active_incidents.append(inc)
                new_incidents.append(inc)
                self.network.close_edge(u, v, current_step)
                logger.debug("Road closure INC%04d on %s->%s", self._counter - 1, u, v)

        # Expire old incidents
        to_remove = []
        for inc in self.active_incidents:
            if current_step >= inc.start_step + inc.duration_steps:
                to_remove.append(inc)
                self.network.update_edge_state(
                    inc.u, inc.v, has_incident=False, step=current_step
                )
                if inc.incident_type == "closure":
                    self.network.reopen_edge(inc.u, inc.v, current_step)
        for inc in to_remove:
            self.active_incidents.remove(inc)

        return new_incidents
