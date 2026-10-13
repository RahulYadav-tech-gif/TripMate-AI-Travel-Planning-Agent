"""Agent orchestrator for TripMate.

Coordinates user input, tool execution via Gemini function calling,
execution time tracking, and response synthesis.
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from google.genai import types

from src.agent.prompts import SYSTEM_PROMPT
from src.config import Settings, get_settings
from src.tools.destination import search_destination_guide
from src.tools.weather import get_weather_forecast

logger = logging.getLogger(__name__)

# Available tools mapped by name
AVAILABLE_TOOLS: Dict[str, Callable[..., Any]] = {
    "search_destination_guide": search_destination_guide,
    "get_weather_forecast": get_weather_forecast,
}


@dataclass
class ToolCallTrace:
    """Record of a single tool execution."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    duration_ms: float
    status: str


@dataclass
class AgentTrace:
    """Full trace of an agent interaction including iterations and tool calls."""

    query: str
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    iterations: int = 0
    final_response: str = ""
    error: Optional[str] = None


class TripMateOrchestrator:
    """Orchestrator coordinating Gemini model turns and tool execution."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.client = client
        if self.client is None and self.settings.validate_api_key():
            try:
                from google import genai
                self.client = genai.Client(api_key=self.settings.gemini_api_key)
                logger.info("Initialized Gemini Client with model %s", self.settings.gemini_model)
            except Exception as exc:
                logger.warning("Could not initialize live Gemini Client: %s", exc)

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Any, float, str]:
        """Executes a tool by name, tracking execution time and catching errors."""
        start_time = time.time()
        func = AVAILABLE_TOOLS.get(tool_name)
        if func is None:
            duration_ms = (time.time() - start_time) * 1000
            err_msg = f"Tool '{tool_name}' is not recognized or supported."
            logger.error(err_msg)
            return {"error": err_msg, "status": "unknown_tool"}, duration_ms, "error"

        try:
            logger.info("[TOOL_EXECUTION] Calling %s with args: %s", tool_name, args)
            result = func(**args)
            duration_ms = (time.time() - start_time) * 1000
            logger.info("[TOOL_RESULT] %s returned in %.1fms (status: success)", tool_name, duration_ms)
            return result, duration_ms, "success"
        except Exception as exc:
            duration_ms = (time.time() - start_time) * 1000
            err_msg = f"Exception during {tool_name} execution: {exc}"
            logger.error("[TOOL_ERROR] %s failed in %.1fms: %s", tool_name, duration_ms, exc)
            return {"error": err_msg, "status": "tool_exception"}, duration_ms, "error"

    def run(self, query: str) -> AgentTrace:
        """Runs the agent loop for a user query."""
        logger.info("[AGENT_START] User query: %r", query)
        trace = AgentTrace(query=query)

        # Validate input
        if not query or not isinstance(query, str) or not query.strip():
            trace.final_response = "Please provide a valid, non-empty travel question."
            logger.warning("[AGENT_END] Empty or malformed query received.")
            return trace

        clean_query = query.strip()

        # Check if live client is available
        if self.client is None:
            err_msg = (
                "Gemini API key is not configured or client failed to initialize. "
                "Please configure GEMINI_API_KEY in .env to run live agent synthesis."
            )
            logger.warning("[AGENT_FALLBACK] %s", err_msg)
            trace.final_response = err_msg
            trace.error = "missing_api_key"
            return trace

        # Set up tools and disable automatic function calling so we control dispatch
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=list(AVAILABLE_TOOLS.values()),
            temperature=0.2,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(timeout=20000),
        )

        contents: List[Any] = [
            types.Content(role="user", parts=[types.Part.from_text(text=clean_query)])
        ]

        active_model = self.settings.gemini_model
        fallback_model = "gemini-flash-lite-latest" if active_model != "gemini-flash-lite-latest" else "gemini-3-flash-preview"

        max_turns = self.settings.max_agent_iterations
        for iteration in range(1, max_turns + 1):
            trace.iterations = iteration
            logger.info("[ITERATION %d/%d] Requesting LLM turn...", iteration, max_turns)

            response = None
            models_to_try = [active_model]
            if fallback_model and fallback_model != active_model:
                models_to_try.append(fallback_model)

            generation_success = False
            last_exc = None

            for current_model in models_to_try:
                max_llm_retries = 3
                for llm_attempt in range(max_llm_retries):
                    try:
                        response = self.client.models.generate_content(
                            model=current_model,
                            contents=contents,
                            config=config,
                        )
                        generation_success = True
                        active_model = current_model
                        break
                    except Exception as exc:
                        last_exc = exc
                        err_str = str(exc).lower()
                        # If primary model hits 429 quota, switch directly to fallback
                        if "429" in err_str or "quota exceeded" in err_str:
                            logger.warning(
                                "Model %s quota reached (%s). Promoting fallback model %s...",
                                current_model,
                                exc,
                                fallback_model,
                            )
                            if fallback_model:
                                active_model = fallback_model
                            break
                        if (
                            ("503" in err_str or "unavailable" in err_str or "high demand" in err_str)
                            and llm_attempt < max_llm_retries - 1
                        ):
                            sleep_s = min(2.0 * (2 ** llm_attempt), 8.0)
                            logger.warning(
                                "Model %s transient spike (%s). Retrying in %.1fs...",
                                current_model,
                                exc,
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                            continue
                        break

                if generation_success:
                    break

            if not generation_success or response is None:
                err_msg = f"LLM generation failed: {last_exc}"
                logger.error("[AGENT_ERROR] %s", err_msg)
                trace.error = "llm_error"
                trace.final_response = f"I encountered an error connecting to the AI model: {last_exc}"
                return trace

            # Check if model requested function calls
            function_calls = response.function_calls
            if function_calls:
                logger.info("[DECISION] LLM requested %d tool call(s): %s",
                            len(function_calls), [fc.name for fc in function_calls])

                # Append model message to history
                if response.candidates and response.candidates[0].content:
                    contents.append(response.candidates[0].content)

                # Execute all requested tool calls
                tool_response_parts: List[types.Part] = []
                for fc in function_calls:
                    call_name = fc.name
                    call_args = dict(fc.args) if fc.args else {}

                    logger.info("[TOOL_SELECTED] Tool: %r | Arguments: %s", call_name, call_args)
                    tool_result, duration_ms, status = self._execute_tool(call_name, call_args)

                    trace.tool_calls.append(
                        ToolCallTrace(
                            tool_name=call_name,
                            arguments=call_args,
                            result=tool_result,
                            duration_ms=round(duration_ms, 1),
                            status=status,
                        )
                    )

                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=call_name,
                            response={"result": tool_result},
                        )
                    )

                # Append tool results to conversation
                contents.append(types.Content(role="user", parts=tool_response_parts))

            else:
                # Model returned final answer
                final_text = response.text or ""
                trace.final_response = final_text.strip()
                logger.info("[SYNTHESIS] Final response generated (%d chars)", len(trace.final_response))
                return trace

        # Exceeded iteration limit
        logger.warning("[AGENT_LIMIT] Agent reached maximum turn limit (%d)", max_turns)
        trace.final_response = (
            f"I was unable to complete the request within the maximum allowed reasoning turns ({max_turns}). "
            "Please try simplifying your question."
        )
        trace.error = "max_iterations_exceeded"
        return trace


def run_agent(query: str, client: Optional[Any] = None) -> AgentTrace:
    """Helper function to run the agent on a single query."""
    orchestrator = TripMateOrchestrator(client=client)
    return orchestrator.run(query)
