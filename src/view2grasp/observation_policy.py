"""One camera policy for every observation phase, including after grasping."""
import numpy as np


def tabletop_view(strategy, initial_view, rng, active_view=None):
    if strategy == "fixed":
        return tuple(initial_view)
    if strategy == "random":
        # Same camera action space as the active agent, including low side views.
        return (float(rng.uniform(-180, 180)), float(rng.uniform(-15, 85)),
                float(rng.uniform(.55, 1.3)))
    if strategy == "active" and active_view is not None:
        return tuple(active_view)
    raise ValueError("active observation requires a VLM-selected view")
