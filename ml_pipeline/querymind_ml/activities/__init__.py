"""Registry of the six ML activities defined by database/neon_demo_setup.sql."""

from __future__ import annotations

from .base import Activity, ActivityResult, ActivitySpec
from .team_productivity_forecast import TeamProductivityForecast
from .tool_adoption_forecast import ToolAdoptionForecast
from .tool_recommendations import ToolRecommendations
from .usage_anomalies import UsageAnomalies
from .user_adoption_risk import UserAdoptionRisk
from .user_segments import UserSegments

ACTIVITIES: dict[str, Activity] = {
    "tool_adoption_forecast": ToolAdoptionForecast(),
    "user_adoption_risk": UserAdoptionRisk(),
    "user_segments": UserSegments(),
    "usage_anomalies": UsageAnomalies(),
    "team_productivity_forecast": TeamProductivityForecast(),
    "tool_recommendations": ToolRecommendations(),
}


def activity_names() -> list[str]:
    return list(ACTIVITIES)


__all__ = [
    "ACTIVITIES",
    "Activity",
    "ActivityResult",
    "ActivitySpec",
    "activity_names",
]
