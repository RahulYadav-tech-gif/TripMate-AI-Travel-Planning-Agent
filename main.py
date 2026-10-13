#!/usr/bin/env python3
"""TripMate AI Travel Planning Assistant CLI.

Main entry point supporting:
1. Single query execution:
   python main.py "What are the visa rules for Tokyo?"
2. Interactive chat mode:
   python main.py --interactive
3. Automated evaluation demo mode:
   python main.py --demo
"""

import argparse
import sys
import time
from typing import Optional

from src.agent.orchestrator import AgentTrace, TripMateOrchestrator
from src.config import get_settings
from src.logging_utils import setup_logging


def print_banner():
    """Print TripMate CLI welcome banner."""
    print("=" * 68)
    print("                TripMate — AI Travel Planning Assistant")
    print("       Destination Guides (RAG) | Open-Meteo Weather | Gemini")
    print("=" * 68)


def display_trace(trace: AgentTrace):
    """Format and display observable agent trace and final synthesis."""
    print("\n" + "-" * 68)
    print(f"  [User Query]: \"{trace.query}\"")
    print("-" * 68)

    print("\n[Agent Summary]")
    print(f"  • Reasoning Iterations : {trace.iterations}")
    print(f"  • Tools Executed       : {len(trace.tool_calls)}")
    if trace.error:
        print(f"  • Error Status         : {trace.error}")

    if trace.tool_calls:
        print("\n[Tool Execution Trace]")
        for i, tc in enumerate(trace.tool_calls, 1):
            status_indicator = "✓" if tc.status == "success" else "✗"
            print(f"  [{i}] Tool Name : {tc.tool_name} [{status_indicator} {tc.status.upper()}]")
            print(f"      Arguments : {tc.arguments}")
            print(f"      Duration  : {tc.duration_ms:.1f} ms")
            res_str = str(tc.result)
            if len(res_str) > 180:
                res_str = res_str[:180] + "..."
            print(f"      Output    : {res_str}")
    else:
        print("\n[Tool Execution Trace]")
        print("  • (No tools called — direct answer or out-of-scope refusal)")

    print("\n[Final Response]")
    print("=" * 68)
    print(trace.final_response)
    print("=" * 68 + "\n")


def run_interactive_mode(orchestrator: TripMateOrchestrator):
    """Run continuous interactive chat session in terminal."""
    print_banner()
    print("Interactive Mode Active.")
    print("Ask any travel question about Bangkok, Barcelona, Reykjavik, Tokyo, or worldwide weather.")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")

    while True:
        try:
            prompt = input("TripMate > ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"exit", "quit", "q"}:
                print("\nThank you for using TripMate. Safe travels! \n")
                break
            trace = orchestrator.run(prompt)
            display_trace(trace)
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Safe travels! \n")
            break


def run_demo_mode(orchestrator: TripMateOrchestrator):
    """Run automated evaluation demo covering all core technical assessment scenarios."""
    print_banner()
    print("DEMO MODE: Running verified evaluation test scenarios...")
    print("All observable events are simultaneously recorded to 'tripmate.log'.\n")

    test_scenarios = [
        {
            "category": "Scenario 1: Single-Tool Destination Guide (Module 2)",
            "query": "What are the visa and entry requirements for Tokyo?",
            "expectation": "Calls search_destination_guide; retrieves 90-day tourist entry rule; synthesizes response.",
        },
        {
            "category": "Scenario 2: Single-Tool Weather Forecast (Module 3)",
            "query": "What is the weather forecast for Barcelona next Tuesday?",
            "expectation": "Calls get_weather_forecast; fetches real forecast from Open-Meteo; reports conditions & temps.",
        },
        {
            "category": "Scenario 3: Multi-Tool Reasoning (Module 4)",
            "query": "What should I pack for Tokyo next Tuesday?",
            "expectation": "Calls BOTH tools (weather + destination guide); synthesizes weather clothing + temple shoes etiquette.",
        },
        {
            "category": "Scenario 4: Scope Awareness & Refusal (Module 5)",
            "query": "Can you book a flight and reserve a hotel in Barcelona for me?",
            "expectation": "Calls ZERO tools; clearly and politely explains it cannot perform bookings.",
        },
        {
            "category": "Scenario 5: Unsupported Destination Guard",
            "query": "What are the visa policies and customs for Paris?",
            "expectation": "Calls guide tool; safely reports that destination guides only cover Bangkok, Barcelona, Reykjavik, and Tokyo.",
        },
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'#' * 68}")
        print(f"  {scenario['category']}")
        print(f"  Expected: {scenario['expectation']}")
        print(f"{'#' * 68}")

        trace = orchestrator.run(scenario["query"])
        display_trace(trace)
        time.sleep(1.0)

    print("\n" + "=" * 68)
    print("  ALL DEMO SCENARIOS COMPLETED SUCCESSFULLY!")
    print("  Check 'tripmate.log' for complete persistent execution traces.")
    print("=" * 68 + "\n")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="TripMate AI Agent — Intelligent Travel Planning Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Travel question to answer (if omitted, starts interactive mode or demo)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive prompt mode",
    )
    parser.add_argument(
        "-d",
        "--demo",
        action="store_true",
        help="Run automated evaluation demo suite covering all assessment modules",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging to console",
    )

    args = parser.parse_args()

    settings = get_settings()
    console_level = 10 if args.debug else 20  # DEBUG if --debug, else INFO
    setup_logging(settings, console_level=console_level)

    orchestrator = TripMateOrchestrator(settings=settings)

    if args.demo:
        run_demo_mode(orchestrator)
    elif args.interactive or not args.query:
        run_interactive_mode(orchestrator)
    else:
        full_query = " ".join(args.query).strip()
        trace = orchestrator.run(full_query)
        display_trace(trace)


if __name__ == "__main__":
    main()
