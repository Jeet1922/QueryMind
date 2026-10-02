from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

ACTIONS = {
    "intent_detection": "INTENT_DETECTION",
    "answer_generation": "ANSWER_GENERATION",
    "insight_summary": "INSIGHT_SUMMARY",
    "visualization": "VISUALIZATION",
}
PROVIDERS = {"openai", "anthropic", "google", "ollama", "groq"}


@dataclass(frozen=True)
class ModelRoute:
    action: str
    provider: str
    model: str
    api_key: str
    base_url: str | None = None

    @property
    def configured(self) -> bool:
        if not self.model or self.provider not in PROVIDERS:
            return False
        if self.provider != "ollama":
            return bool(self.api_key)
        base_url = self.base_url or "http://localhost:11434/v1"
        local_host = urlsplit(base_url).hostname in {"localhost", "127.0.0.1", "::1"}
        return local_host or bool(self.api_key)


class LLMRouter:
    """Routes bounded language tasks to configurable providers; SQL stays registry-controlled."""

    def __init__(self, environ: dict[str, str] | None = None, timeout_seconds: float = 20.0):
        self.environ = os.environ if environ is None else environ
        self.timeout_seconds = timeout_seconds

    def get_route(self, action: str) -> ModelRoute:
        if action not in ACTIONS:
            raise ValueError(f"Unsupported LLM action: {action}")
        prefix = f"INSIGHTMESH_{ACTIONS[action]}_"
        generic_provider = self.environ.get("INSIGHTMESH_LLM_PROVIDER", "openai").lower()
        generic_model = self.environ.get("INSIGHTMESH_LLM_MODEL", "")
        provider = (self.environ.get(prefix + "PROVIDER") or generic_provider).lower()
        api_key = (
            self.environ.get("GROQ_API_KEY", "")
            if provider == "groq"
            else self.environ.get(prefix + "API_KEY") or self.environ.get("INSIGHTMESH_LLM_API_KEY", "")
        )
        return ModelRoute(
            action=action,
            provider=provider,
            model=self.environ.get(prefix + "MODEL") or generic_model,
            api_key=api_key,
            base_url=self.environ.get(prefix + "BASE_URL") or self.environ.get("INSIGHTMESH_LLM_BASE_URL"),
        )

    def status(self) -> list[dict[str, str | bool]]:
        return [
            {
                "action": action,
                "provider": route.provider,
                "model": route.model or "not configured",
                "configured": route.configured,
            }
            for action in ACTIONS
            if (route := self.get_route(action))
        ]

    def complete(self, action: str, system_prompt: str, user_prompt: str) -> str | None:
        route = self.get_route(action)
        if not route.configured:
            return None
        try:
            if route.provider in {"openai", "ollama", "groq"}:
                payload = self._complete_openai(route, system_prompt, user_prompt)
            elif route.provider == "anthropic":
                payload = self._complete_anthropic(route, system_prompt, user_prompt)
            else:
                payload = self._complete_google(route, system_prompt, user_prompt)
            return payload.strip() or None
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("LLM request failed for action %s (%s)", action, type(exc).__name__)
            return None

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _complete_openai(self, route: ModelRoute, system_prompt: str, user_prompt: str) -> str:
        default_url = {
            "ollama": "http://localhost:11434/v1",
            "groq": "https://api.groq.com/openai/v1",
        }.get(route.provider, "https://api.openai.com/v1")
        base_url = (route.base_url or default_url).rstrip("/")
        headers = {"Authorization": f"Bearer {route.api_key}"} if route.api_key else {}
        response = self._post_json(
            f"{base_url}/chat/completions",
            headers,
            {
                "model": route.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
            },
        )
        return response["choices"][0]["message"]["content"]

    def _complete_anthropic(self, route: ModelRoute, system_prompt: str, user_prompt: str) -> str:
        base_url = (route.base_url or "https://api.anthropic.com/v1").rstrip("/")
        response = self._post_json(
            f"{base_url}/messages",
            {"x-api-key": route.api_key, "anthropic-version": "2023-06-01"},
            {
                "model": route.model,
                "max_tokens": 700,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
        return "".join(block["text"] for block in response["content"] if block.get("type") == "text")

    def _complete_google(self, route: ModelRoute, system_prompt: str, user_prompt: str) -> str:
        base_url = (route.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        model = quote(route.model, safe="-")
        response = self._post_json(
            f"{base_url}/models/{model}:generateContent?key={quote(route.api_key, safe='')}",
            {},
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
        )
        return "".join(part["text"] for part in response["candidates"][0]["content"]["parts"] if "text" in part)

    def generate_answer(self, question: str, intent: str, rows: list[dict[str, Any]]) -> str | None:
        return self.complete(
            "answer_generation",
            "Answer in concise, plain English using only the supplied analytics records. State when values are forecasts or model estimates. Do not make causal claims. Do not invent values.",
            json.dumps({"question": question, "intent": intent, "records": rows[:20]}, default=str),
        )

    def summarize_insight(self, question: str, intent: str, rows: list[dict[str, Any]]) -> str | None:
        return self.complete(
            "insight_summary",
            "Write one brief decision-oriented insight in English, grounded only in the supplied records. Do not imply causation or invent numbers.",
            json.dumps({"question": question, "intent": intent, "records": rows[:20]}, default=str),
        )

    def recommend_visualization(self, question: str, rows: list[dict[str, Any]]) -> str | None:
        recommendation = self.complete(
            "visualization",
            "Choose the clearest visualization for this tabular data. Return exactly one value: bar, horizontal_bar, line, donut, or table. Never return anything else.",
            json.dumps({"question": question, "records": rows[:10]}, default=str),
        )
        allowed = {"bar", "horizontal_bar", "line", "donut", "table"}
        if recommendation:
            normalized = recommendation.strip().lower().replace("`", "")
            if normalized in allowed:
                return normalized
        return None

    def classify_intent(self, question: str, fallback_intent: str) -> str:
        allowed = [
            "TOOL_RANKING",
            "ADOPTION_FORECAST",
            "ADOPTION_RISK",
            "USER_SEGMENTATION",
            "USAGE_ANOMALY",
            "PRODUCTIVITY_FORECAST",
            "DEPARTMENT_ANALYSIS",
            "OUT_OF_SCOPE",
        ]
        response = self.complete(
            "intent_detection",
            "Classify the analytics request. Return exactly one JSON object with an intent key. Allowed values: " + ", ".join(allowed) + ". Use OUT_OF_SCOPE for requests unrelated to enterprise analytics. Do not infer unsupported data needs.",
            question,
        )
        if not response:
            return fallback_intent
        try:
            predicted = json.loads(response).get("intent")
        except (json.JSONDecodeError, AttributeError):
            return fallback_intent
        return predicted if predicted in allowed else fallback_intent
