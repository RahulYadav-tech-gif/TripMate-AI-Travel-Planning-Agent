"""Integration tests for the TripMate agent.

Tests multi-tool flows, RAG integration, horizon guards, and live API pipeline.
"""

from unittest.mock import MagicMock
import pytest

from src.agent.orchestrator import TripMateOrchestrator, run_agent
from src.config import get_settings


def create_mock_function_call(name: str, args: dict):
    """Construct mock FunctionCall object."""
    fc = MagicMock()
    fc.name = name
    fc.args = args
    return fc


def create_mock_response(text: str = "", function_calls=None):
    """Construct mock GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    resp.function_calls = function_calls or []
    candidate = MagicMock()
    candidate.content = MagicMock()
    resp.candidates = [candidate]
    return resp


def test_full_multi_tool_flow_end_to_end():
    """Integration Test 1: Full multi-tool flow end-to-end with real tool execution.

    Simulates the model requesting both tools in Turn 1, verifies that BOTH
    actual tools execute against the real knowledge base and weather parser,
    and checks that Turn 2 produces the synthesized output.
    """
    mock_client = MagicMock()

    # Turn 1: Model requests both real tools
    fc_weather = create_mock_function_call(
        "get_weather_forecast",
        {"city": "Tokyo", "date_or_month": "tomorrow"},
    )
    fc_guide = create_mock_function_call(
        "search_destination_guide",
        {"query": "Tokyo packing tips and etiquette"},
    )
    turn1_resp = create_mock_response(text="", function_calls=[fc_weather, fc_guide])

    # Turn 2: Synthesize answer
    turn2_resp = create_mock_response(
        text=(
            "Based on the weather in Tokyo tomorrow and local cultural customs, "
            "bring layers and comfortable slip-on walking shoes for visiting shrines."
        )
    )
    mock_client.models.generate_content.side_effect = [turn1_resp, turn2_resp]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("What should I pack for Tokyo tomorrow?")

    # Assertions on observable trace
    assert trace.iterations == 2
    assert len(trace.tool_calls) == 2

    # Verify real weather tool executed and returned valid structure
    weather_trace = next(t for t in trace.tool_calls if t.tool_name == "get_weather_forecast")
    assert weather_trace.status == "success"
    assert weather_trace.duration_ms > 0
    assert "conditions" in weather_trace.result or "status" in weather_trace.result

    # Verify real RAG search executed against local ChromaDB
    guide_trace = next(t for t in trace.tool_calls if t.tool_name == "search_destination_guide")
    assert guide_trace.status == "success"
    assert guide_trace.duration_ms > 0
    assert len(guide_trace.result) > 0  # Retrieved real chunks from tokyo.txt
    assert any("tokyo" in chunk.lower() for chunk in guide_trace.result)

    # Verify final synthesis
    assert "shoes" in trace.final_response.lower()
    assert trace.error is None


def test_destination_guide_single_tool_integration():
    """Integration Test 2: Single-tool Destination Guide flow executing real RAG."""
    mock_client = MagicMock()

    fc = create_mock_function_call("search_destination_guide", {"query": "Barcelona local customs"})
    resp1 = create_mock_response(text="", function_calls=[fc])
    resp2 = create_mock_response(text="In Barcelona, dinner is typically eaten late, around 9:00 PM or 10:00 PM.")
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("What are the dining customs in Barcelona?")

    assert trace.iterations == 2
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "search_destination_guide"
    assert trace.tool_calls[0].status == "success"
    # Ensure real chunks were retrieved from barcelona.txt in ChromaDB
    assert len(trace.tool_calls[0].result) >= 1
    assert any("barcelona" in chunk.lower() for chunk in trace.tool_calls[0].result)
    assert "dinner" in trace.final_response.lower()


def test_weather_horizon_guard_integration():
    """Integration Test 3: Weather tool handling beyond-15-day horizon in agent loop."""
    mock_client = MagicMock()

    fc = create_mock_function_call("get_weather_forecast", {"city": "Reykjavik", "date_or_month": "December"})
    resp1 = create_mock_response(text="", function_calls=[fc])
    resp2 = create_mock_response(
        text="Real meteorological forecasts are only available up to 15 days in advance."
    )
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("What is the weather in Reykjavik in December?")

    assert trace.iterations == 2
    assert len(trace.tool_calls) == 1
    tool_call = trace.tool_calls[0]
    assert tool_call.tool_name == "get_weather_forecast"
    assert tool_call.result.get("status") == "unavailable"
    assert "15 days" in tool_call.result.get("message")
    assert "15 days" in trace.final_response


def test_live_end_to_end_agent_flow():
    """Integration Test 4: Live end-to-end integration test with live Gemini API & Open-Meteo.

    Executes full live pipeline if GEMINI_API_KEY is configured in the environment.
    """
    settings = get_settings()
    if not settings.validate_api_key():
        pytest.skip("Skipping live integration test: GEMINI_API_KEY is not set.")

    orchestrator = TripMateOrchestrator(settings=settings)
    trace = orchestrator.run("What is the weather in Tokyo tomorrow?")

    assert trace.error is None
    assert trace.iterations >= 1
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "get_weather_forecast"
    assert trace.tool_calls[0].status == "success"
    assert len(trace.final_response) > 20
    assert "tokyo" in trace.final_response.lower()
