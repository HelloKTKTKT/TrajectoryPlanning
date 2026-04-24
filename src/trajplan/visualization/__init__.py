from trajplan.visualization.messages import (
    AgentVisualizationSnapshot,
    PlannerVisualizationSnapshot,
)


def __getattr__(name):
    if name == "LiveVisualizer":
        from trajplan.visualization.live_visualizer import LiveVisualizer
        return LiveVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentVisualizationSnapshot",
    "PlannerVisualizationSnapshot",
    "LiveVisualizer",
]
