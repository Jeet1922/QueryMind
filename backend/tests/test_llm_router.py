from backend.app.llm.router import LLMRouter


def test_task_routes_accept_independent_providers_and_models():
    router = LLMRouter(environ={
        "INSIGHTMESH_INTENT_DETECTION_PROVIDER": "openai",
        "INSIGHTMESH_INTENT_DETECTION_MODEL": "intent-model",
        "INSIGHTMESH_INTENT_DETECTION_API_KEY": "intent-key",
        "INSIGHTMESH_ANSWER_GENERATION_PROVIDER": "anthropic",
        "INSIGHTMESH_ANSWER_GENERATION_MODEL": "answer-model",
        "INSIGHTMESH_ANSWER_GENERATION_API_KEY": "answer-key",
    })

    status = {route["action"]: route for route in router.status()}
    assert status["intent_detection"]["model"] == "intent-model"
    assert status["intent_detection"]["configured"] is True
    assert status["answer_generation"]["provider"] == "anthropic"
    assert status["answer_generation"]["model"] == "answer-model"
    assert status["answer_generation"]["configured"] is True
    assert status["insight_summary"]["configured"] is False


def test_intent_route_rejects_unrecognized_model_output():
    router = LLMRouter(environ={})
    router.complete = lambda *args: '{"intent":"RUN_ARBITRARY_SQL"}'

    assert router.classify_intent("show adoption", "TOOL_RANKING") == "TOOL_RANKING"


def test_visualization_route_accepts_only_supported_chart_types():
    router = LLMRouter(environ={})
    router.complete = lambda *args: "pie"

    assert router.recommend_visualization("show adoption", [{"rate": 0.5}]) is None


def test_local_ollama_route_does_not_require_an_api_key():
    router = LLMRouter(environ={
        "INSIGHTMESH_LLM_PROVIDER": "ollama",
        "INSIGHTMESH_ANSWER_GENERATION_MODEL": "qwen3.8",
    })

    route = router.get_route("answer_generation")
    assert route.configured is True
    assert route.api_key == ""


def test_ollama_cloud_route_uses_shared_api_key_and_base_url():
    router = LLMRouter(environ={
        "INSIGHTMESH_LLM_PROVIDER": "ollama",
        "INSIGHTMESH_LLM_BASE_URL": "https://ollama.com/v1",
        "INSIGHTMESH_LLM_API_KEY": "cloud-key",
        "INSIGHTMESH_ANSWER_GENERATION_MODEL": "qwen3.8",
        "INSIGHTMESH_ANSWER_GENERATION_API_KEY": "",
    })

    route = router.get_route("answer_generation")
    assert route.configured is True
    assert route.api_key == "cloud-key"
    assert route.base_url == "https://ollama.com/v1"


def test_groq_route_uses_groq_key_and_openai_compatible_endpoint():
    router = LLMRouter(environ={
        "INSIGHTMESH_LLM_PROVIDER": "groq",
        "INSIGHTMESH_ANSWER_GENERATION_MODEL": "llama-3.3-70b-versatile",
        "GROQ_API_KEY": "test-groq-key",
        "INSIGHTMESH_LLM_API_KEY": "old-ollama-key",
    })
    captured = {}

    def fake_post(url, headers, payload):
        captured.update(url=url, headers=headers, payload=payload)
        return {"choices": [{"message": {"content": "READY"}}]}

    router._post_json = fake_post
    route = router.get_route("answer_generation")
    assert route.configured is True
    assert router._complete_openai(route, "system", "user") == "READY"
    assert captured["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-groq-key"
