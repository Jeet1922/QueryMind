from __future__ import annotations

PROMPT_INJECTION_PATTERNS = {
    "ignore your instructions",
    "reveal hidden",
    "system prompt",
    "developer prompt",
    "override instructions",
    "bypass security",
    "ignore system",
    "reveal the prompt",
    "modifying application behavior",
    "change the application",
    "ignore prior instructions",
}

OUT_OF_DOMAIN_PATTERNS = {
    "weather",
    "rain",
    "sunny",
    "temperature",
    "stock price",
    "football score",
    "latest news",
    "travel booking",
}

SENSITIVE_PATTERNS = {
    "password",
    "token",
    "secret",
    "api key",
    "credentials",
    "private key",
}


def validate_user_question(question: str) -> str:
    if question is None or not str(question).strip():
        raise ValueError("Question is empty.")

    normalized = str(question).strip()
    lower = normalized.lower()

    if any(pattern in lower for pattern in PROMPT_INJECTION_PATTERNS):
        raise ValueError("Prompt injection attempt rejected.")

    if any(pattern in lower for pattern in SENSITIVE_PATTERNS):
        raise ValueError("Sensitive-data requests are not allowed.")

    if any(pattern in lower for pattern in OUT_OF_DOMAIN_PATTERNS) and "adoption" not in lower and "usage" not in lower:
        raise ValueError("Question is outside the approved analytics scope.")

    if any(keyword in lower for keyword in ("ignore", "reveal", "override")) and any(keyword in lower for keyword in ("prompt", "instruction", "system")):
        raise ValueError("Prompt injection attempt rejected.")

    return normalized
