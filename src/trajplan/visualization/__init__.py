from trajplan.visualization.messages import (
    AgentVisualizationSnapshot,
    PlannerVisualizationSnapshot,
)

__all__ = [
    "AgentVisualizationSnapshot",
    "PlannerVisualizationSnapshot",
    "LiveVisualizer",
]


def __getattr__(name: str) -> object:
    if name == "LiveVisualizer":
        from trajplan.visualization.live_visualizer import LiveVisualizer
        return LiveVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
