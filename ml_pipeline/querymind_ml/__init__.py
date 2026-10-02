"""QueryMind ML pipeline package.

This package is the separate ML/MLOps project referenced by the repository
README: the application itself only consumes persisted outputs from the ml_*
tables, while training, evaluation, registration and monitoring live here.
"""

__all__ = ["config", "db", "registry", "tracking", "monitor", "sql_runner"]
