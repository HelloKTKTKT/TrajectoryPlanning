from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Queue

from trajplan.planning.messages import PlannerTickInput, PlannerTickOutput


@dataclass(slots=True)
class PlannerManagerAgentQueues:
    pm_to_agent: Queue[PlannerTickOutput]
    agent_to_pm: Queue[PlannerTickInput]
