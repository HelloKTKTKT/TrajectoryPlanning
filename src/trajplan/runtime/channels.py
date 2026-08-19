from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Queue

from trajplan.planning.messages import (
    NeighborTrajectoryMessage,
    PlannerTickInput,
    PlannerTickOutput,
)
from trajplan.visualization.messages import (
    AgentVisualizationSnapshot,
    GoalUpdateMessage,
    ObstacleUpdateMessage,
    PlannerVisualizationSnapshot,
)


@dataclass(slots=True)
class PlannerManagerAgentQueues:
    pm_to_agent: Queue[PlannerTickOutput]
    agent_to_pm: Queue[PlannerTickInput]


@dataclass(slots=True)
class VisualizationQueues:
    planner_to_visualizer: Queue[PlannerVisualizationSnapshot]
    agent_to_visualizer: Queue[AgentVisualizationSnapshot]


@dataclass(slots=True)
class PlannerSwarmQueues:
    planner_to_relay: Queue[NeighborTrajectoryMessage]
    relay_to_planner: Queue[NeighborTrajectoryMessage]


@dataclass(slots=True)
class ControlQueues:
    gui_to_planner: Queue[GoalUpdateMessage | ObstacleUpdateMessage]
