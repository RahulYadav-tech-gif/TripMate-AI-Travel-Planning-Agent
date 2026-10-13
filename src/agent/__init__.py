"""TripMate Agent Core Package.

Exposes the main agent orchestrator and trace dataclasses.
"""

from src.agent.orchestrator import (
    TripMateOrchestrator,
    AgentTrace,
    ToolCallTrace,
    run_agent,
)
from src.agent.prompts import SYSTEM_PROMPT

__all__ = [
    "TripMateOrchestrator",
    "AgentTrace",
    "ToolCallTrace",
    "run_agent",
    "SYSTEM_PROMPT",
]
