"""Command-line interface for aggfly.

The CLI is a thin orchestrator over the public ``aggfly`` API: it parses a YAML
config, resolves preprocessing, and calls the same ``af.*`` functions a user
would call from a script. See ``internal/cli-plan.md`` for the full design.
"""

from .main import cli

__all__ = ["cli"]
