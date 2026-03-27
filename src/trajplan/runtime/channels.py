from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Queue


@dataclass(slots=True)
class PlannerManagerAgentQueues:
    pm_to_agent: Queue
    agent_to_pm: Queue


# @dataclass(slots=True)
# class PlannerManagerMainQueues:
#     pm_to_main: Queue
#     main_to_pm: Queue
