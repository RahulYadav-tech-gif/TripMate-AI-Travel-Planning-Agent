AI Agent Developer – Technical Assessment
Experience Level: 1 – 1.5 Years
Assignment Duration: 3–5 Days (expected effort: a few focused hours, not full-time work)
Submission: GitHub Repository (public/private), along with a README, example trace logs, and a short Loom video

---

## Objective

This assessment is designed to evaluate your ability to:
● Design and reason about agentic AI systems.
● Build tools that an LLM-based agent can call dynamically based on user intent.
● Integrate Retrieval-Augmented Generation (RAG) into an agentic workflow.
● Make appropriate technical decisions and justify them.
● Write readable, testable, and modular code.

---

## Business Scenario

A travel company wants to launch "TripMate" — an AI assistant that helps travelers plan trips by answering natural-language questions about destinations: visa requirements, weather, packing advice, safety, and local customs.

Currently, this information is scattered across static FAQ pages and requires manual research across multiple sources. Your task is to build the **agentic core** of TripMate: an AI agent that can reason about a user's request and dynamically decide which tool(s) it needs to answer it — rather than following a fixed script or keyword-based routing.

---

## Functional Requirements

### Module 1 – Agent Core / Orchestration

The agent should:
1. Accept a natural-language user query.
2. Dynamically decide — per query — which tool(s), if any, are relevant.
3. Call the selected tool(s), in the correct order if multiple are needed.
4. Synthesize a single, coherent natural-language response from the tool output(s).
5. Produce a visible reasoning trace (which tool(s) were called, with what arguments, and why) — via logging, verbose mode, or an equivalent mechanism.

### Module 2 – Destination Knowledge Tool (RAG)

`search_destination_guide(query: str) → List[str]`

- Input: a natural-language question or topic.
- Output: the most relevant chunk(s) of text from a destination knowledge base.
- A destination data pack (visa/entry notes, best time to visit, local customs, packing tips, safety notes for 4 cities) will be provided for you to ingest — you do not need to source your own content.
- You are free to choose your own embedding model, chunking strategy, and vector store (or an in-memory similarity search).

### Module 3 – Weather Forecast Tool

`get_weather_forecast(city: str, date_or_month: str) → dict`

- Input: a city name and a date or month reference.
- Output: e.g. `{ "conditions": "cold, occasional snow", "temp_range_c": [-3, 3] }` — exact schema is your choice, just be consistent.
- **Implementation is your choice:**
  - **Option A:** call any real, free/keyless weather API (e.g. Open-Meteo).
  - **Option B:** build a small hardcoded/mock lookup table by city and month.
  - Neither option is scored differently — choose whichever lets you focus your time on the agentic logic rather than API integration.

### Module 4 – Multi-Tool Reasoning

The agent must correctly handle at least one category of query that requires **both tools together** in a single response — the clearest example is a packing question, since good packing advice should reflect both the destination guide's tips and the actual weather/season for that time of year.

### Module 5 – Scope Awareness

The agent must handle requests outside its capabilities (e.g. "can you book my flight?") by clearly stating it cannot do this, rather than fabricating an action or result.

---

## Technical Requirements

Use:
● Python 3.10+
● Any agent framework of your choice — LangChain, LangGraph, CrewAI, a raw OpenAI/Anthropic function-calling loop, or your own custom orchestration.
● Any LLM provider you have access to (OpenAI, Anthropic, an open-source model via Ollama, etc.).
● Any vector store — FAISS, ChromaDB, or in-memory similarity search.
● A free/keyless weather API of your choice, or a mock dataset (see Module 3).
● Git

---

## Non-Functional Requirements

Your solution should demonstrate:
● Clean, layered design — separation between tool definitions, orchestration/agent logic, and any business logic.
● Configuration management (e.g. API keys via environment variables, not hardcoded).
● Structured logging of tool calls and reasoning.
● Exception handling around tool execution.
● Input validation.
● Modular, readable code structure.

We are **not** evaluating production-readiness, UI polish, or lines of code — a small, clean, well-reasoned solution is valued far more than a large one.

---

## Error Handling

Handle scenarios such as:
● Unknown/unsupported destination.
● Missing or incomplete weather data for a given city/date.
● Ambiguous user query (unclear which tool, if any, applies).
● Tool failure or timeout (e.g. a real weather API call fails).
● Out-of-scope request (e.g. booking, unrelated topics).
● Malformed or empty input.

Return clear, meaningful responses in each case rather than silent failures or fabricated answers.

---

## Logging

Implement structured logging for:
● Each tool call (tool name, arguments, result).
● The agent's reasoning/decision trace.
● Errors and fallback behavior.

---

## Testing

Include:
● Unit tests for each individual tool.
● Tests validating correct tool selection for a set of sample queries (single-tool, multi-tool, no-tool).
● At least one integration test demonstrating a full multi-tool flow end-to-end.

---

## Documentation

Your README should include:
● Project setup and run instructions.
● Architecture overview (see below).
● Tool schemas/descriptions as given to the LLM.
● 3–4 example runs showing the full trace (input → tool call(s) → output), including at least one multi-tool example and the out-of-scope example.
● Design decisions and assumptions.
● Known limitations.
● Suggested future improvements.

---

## Architecture

Provide a simple architecture diagram showing:
● The Agent / Orchestrator layer.
● The RAG tool and its vector store.
● The Weather Forecast tool.
● The LLM provider.
● How a request flows through these components and back to the user.

---

## Scalability Considerations (Discussion Only — No Implementation Required)

Briefly address in your README:
● How your RAG tool would scale if the knowledge base grew from 4 cities to several hundred.
● How you would avoid redundant tool calls or LLM calls for repeated/similar queries.
● How you would reduce LLM API costs at higher query volume.
● How you would keep tool-selection latency low as the number of available tools grows.

---

## Loom Video Requirements (Mandatory, 5–8 minutes)

The video should cover:
1. **Overview** — the problem statement and your overall approach.
2. **Architecture Walkthrough** — explain your diagram and how components communicate.
3. **Agent Demo** — run a few example queries live, showing single-tool, multi-tool (chained), and out-of-scope handling.
4. **Code Overview** — walk through your tool definitions and orchestration logic.
5. **Technical Decisions** — why you chose your framework, LLM provider, vector store, and weather tool approach.
6. **Limitations & Future Improvements** — what you'd change or add with more time.

---

## Final Deliverables

● GitHub Repository
● README (as specified above)
● Architecture Diagram
● Destination data pack ingested into your RAG tool (provided to you separately)
● Test Cases
● Loom Video (Mandatory)
