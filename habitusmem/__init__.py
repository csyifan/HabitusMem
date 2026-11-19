"""HabitusMem: belief-state memory for personalized LLM agents."""

__version__ = "0.1.0"

__all__ = ["BeliefWriter", "with_execution_guidance"]


def __getattr__(name):
    if name == "BeliefWriter":
        from .writer import BeliefWriter

        return BeliefWriter
    if name == "with_execution_guidance":
        from .prompts import with_execution_guidance

        return with_execution_guidance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
