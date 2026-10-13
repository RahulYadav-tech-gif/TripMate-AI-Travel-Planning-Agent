"""Tests for agent tool selection and decision logic.

Covers single-tool, multi-tool, out-of-scope refusal, and edge cases.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.agent.orchestrator import TripMateOrchestrator, run_agent
from src.config import Settings


def create_mock_function_call(name: str, args: dict):
    """Helper to construct a mock FunctionCall object."""
    fc = MagicMock()
    fc.name = name
    fc.args = args
    return fc


def create_mock_response(text: str = "", function_calls=None):
    """Helper to construct a mock GenerateContentResponse."""
    resp = MagicMock()
    resp.text = text
    resp.function_calls = function_calls or []
    candidate = MagicMock()
    candidate.content = MagicMock()
    resp.candidates = [candidate]
    return resp


def test_single_tool_selection_destination_guide():
    """Verify single-tool selection for Destination Guide query (Tokyo visa rules)."""
    mock_client = MagicMock()

    # Turn 1: Model decides to call search_destination_guide
    fc = create_mock_function_call("search_destination_guide", {"query": "Tokyo visa rules"})
    resp1 = create_mock_response(text="", function_calls=[fc])

    # Turn 2: Model synthesizes final response based on tool output
    resp2 = create_mock_response(
        text="Many nationalities can visit Tokyo visa-free for tourism up to 90 days."
    )
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("What are the visa rules for Tokyo?")

    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "search_destination_guide"
    assert trace.tool_calls[0].status == "success"
    assert "visa-free" in trace.final_response
    assert trace.iterations == 2


def test_single_tool_selection_weather_forecast():
    """Verify single-tool selection for Weather query (Bangkok weather tomorrow)."""
    mock_client = MagicMock()

    # Turn 1: Model decides to call get_weather_forecast
    fc = create_mock_function_call("get_weather_forecast", {"city": "Bangkok", "date_or_month": "tomorrow"})
    resp1 = create_mock_response(text="", function_calls=[fc])

    # Turn 2: Model synthesizes final response
    resp2 = create_mock_response(
        text="Tomorrow in Bangkok, expect warm weather around 30°C."
    )
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("What is the weather in Bangkok tomorrow?")

    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "get_weather_forecast"
    assert trace.tool_calls[0].status == "success"
    assert "Bangkok" in trace.final_response
    assert trace.iterations == 2


def test_multi_tool_reasoning_packing_advice():
    """Verify Module 4: Multi-tool reasoning calling both tools for packing advice."""
    mock_client = MagicMock()

    # Turn 1: Model requests both get_weather_forecast AND search_destination_guide
    fc_weather = create_mock_function_call("get_weather_forecast", {"city": "Tokyo", "date_or_month": "next Tuesday"})
    fc_guide = create_mock_function_call("search_destination_guide", {"query": "Tokyo packing tips"})
    resp1 = create_mock_response(text="", function_calls=[fc_weather, fc_guide])

    # Turn 2: Model synthesizes unified answer combining weather and cultural attire
    resp2 = create_mock_response(
        text=(
            "For Tokyo next Tuesday, expect mild temperatures around 18°C. "
            "Pack layers and bring comfortable slip-on walking shoes for visiting shrines."
        )
    )
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("What should I pack for Tokyo next Tuesday?")

    # Verify both tools were executed
    assert len(trace.tool_calls) == 2
    tool_names = {t.tool_name for t in trace.tool_calls}
    assert tool_names == {"get_weather_forecast", "search_destination_guide"}
    assert all(t.status == "success" for t in trace.tool_calls)

    # Verify synthesis includes both components
    assert "18°C" in trace.final_response or "mild" in trace.final_response
    assert "shoes" in trace.final_response or "layers" in trace.final_response


def test_no_tool_out_of_scope_flight_booking():
    """Verify Module 5: Scope awareness - politely refuses flight booking without calling tools."""
    mock_client = MagicMock()

    # Model immediately returns polite refusal without calling any tools
    resp = create_mock_response(
        text=(
            "I cannot book flights or make reservations, as I am an informational travel assistant. "
            "However, I can help you with visa requirements, weather forecasts, or packing advice for Tokyo!"
        ),
        function_calls=[],
    )
    mock_client.models.generate_content.return_value = resp

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("Can you book my flight to Tokyo?")

    assert len(trace.tool_calls) == 0
    assert "cannot book" in trace.final_response.lower()
    assert trace.iterations == 1


def test_unsupported_destination_guidance():
    """Verify handling of unsupported destination for guide search."""
    mock_client = MagicMock()

    fc = create_mock_function_call("search_destination_guide", {"query": "Sydney visa policy"})
    resp1 = create_mock_response(text="", function_calls=[fc])

    resp2 = create_mock_response(
        text=(
            "TripMate currently only has destination guides for Bangkok, Barcelona, Reykjavik, and Tokyo. "
            "I do not have guide information for Sydney."
        )
    )
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("What is the visa policy for Sydney?")

    assert len(trace.tool_calls) == 1
    # The indexed knowledge base contains only 4 cities, never Sydney
    for chunk in trace.tool_calls[0].result:
        assert "sydney" not in chunk.lower()
    assert "Bangkok, Barcelona, Reykjavik, and Tokyo" in trace.final_response


def test_empty_and_whitespace_query_validation():
    """Verify empty queries return instant validation message without LLM calls."""
    mock_client = MagicMock()
    orchestrator = TripMateOrchestrator(client=mock_client)

    trace_empty = orchestrator.run("")
    assert "valid" in trace_empty.final_response.lower()
    assert len(trace_empty.tool_calls) == 0
    mock_client.models.generate_content.assert_not_called()

    trace_spaces = orchestrator.run("    ")
    assert "valid" in trace_spaces.final_response.lower()
    assert len(trace_spaces.tool_calls) == 0


def test_tool_execution_exception_handling():
    """Verify orchestrator catches tool execution exceptions gracefully."""
    mock_client = MagicMock()

    # Model requests unknown tool
    fc_bad = create_mock_function_call("non_existent_tool", {"param": "val"})
    resp1 = create_mock_response(text="", function_calls=[fc_bad])

    resp2 = create_mock_response(text="I apologize, but that tool was unavailable.")
    mock_client.models.generate_content.side_effect = [resp1, resp2]

    orchestrator = TripMateOrchestrator(client=mock_client)
    trace = orchestrator.run("Trigger unknown tool")

    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].status == "error"
    assert "error" in trace.tool_calls[0].result


def test_max_iterations_loop_safety():
    """Verify orchestrator terminates when loop exceeds maximum configured iterations."""
    mock_client = MagicMock()

    # Model continuously calls tool without ever generating final text
    fc = create_mock_function_call("search_destination_guide", {"query": "Tokyo"})
    infinite_resp = create_mock_response(text="", function_calls=[fc])
    mock_client.models.generate_content.return_value = infinite_resp

    test_settings = Settings(max_agent_iterations=3)
    orchestrator = TripMateOrchestrator(settings=test_settings, client=mock_client)
    trace = orchestrator.run("Keep looping")

    assert trace.iterations == 3
    assert trace.error == "max_iterations_exceeded"
    assert "maximum" in trace.final_response.lower()
