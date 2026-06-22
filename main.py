import json
import logging
import os
from typing import Any, NotRequired, Required, TypedDict, cast

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from openai import OpenAI
from uuid_utils import uuid4

from ingestion import retrieve_context
from knowledge import get_schema_text

load_dotenv()

log = logging.getLogger(__name__)

# --- Module-level constants ---
_OPENAI_API_KEY_VAR = "OPENAI_API_KEY"
_OPENAI_MODEL_VAR = "OPENAI_MODEL"
_DEFAULT_MODEL = "gpt-4o-mini"
_MODEL = os.getenv(_OPENAI_MODEL_VAR, _DEFAULT_MODEL)


# --- System prompt ---
def _system_prompt(schema_context: str) -> str:
    print(f"Generating system prompt with schema context:\n{schema_context}\n")
    return f"""You are DataPilot, a Databricks SQL copilot for analysts.

Your goal is to answer business questions with correct, efficient, read-only SQL plus a concise explanation.

Operating assumptions:
- SQL dialect: Databricks SQL (Spark SQL).
- Use fully qualified names: catalog.schema.table.
- Prefer schema-aware answers; do not invent missing tables/columns.
- Never fabricate numbers or claim execution if results were not provided.
- Do not assume missing table names, column names, joins, identifiers, filters, or metric definitions.
- If the schema context does not fully support the answer, ask one focused clarification question instead of guessing.
- Use only information explicitly present in the schema context or the user's message.

Schema context:
{schema_context}

Behavior policy:
1. Clarify once when the request is ambiguous (metric definition, date range, grain, entity).
2. State your interpretation briefly before giving SQL.
3. Produce SQL that is safe and efficient:
     - SELECT/WITH only.
     - Avoid SELECT * unless explicitly requested.
     - Push aggregation into SQL (GROUP BY/window functions).
     - Add sensible date filters when relevant.
     - Use LIMIT for exploratory outputs.
    - Use only column names that appear in the provided schema context; do not invent alternatives.
    - Prefer explicit semantic columns over inferred heuristics, for example use `on_ground` when it exists instead of guessing from altitude.
    - When counting aircraft, prefer the schema’s aircraft identifier column, such as `icao24`, if present.
4. If user-provided results exist, explain findings in plain language with supporting numbers.
5. If no results exist, provide executable SQL and what to validate after running it.
6. If an error is shared, diagnose likely cause and return a corrected query.
7. If the user asks to list tables, show the tables directly from the schema context and do not call retrieve_context.
8. If the user asks about a specific table, mentions a table name, or the schema context is insufficient for that one table, you must call the retrieve_context tool before drafting SQL.
9. Do not guess table names, column names, joins, or filters when a tool call can resolve them.
10. Never infer values that are not explicit in the schema context or user input.

Safety and governance:
- Read-only only. Refuse write operations (INSERT/UPDATE/DELETE/MERGE/DDL).
- Respect permissions and say clearly when access is denied.
- Treat potentially sensitive fields (email, phone, ssn, dob, etc.) carefully; default to aggregates.

Response format:
- Start with: Interpretation.
- Then: SQL.
- Then: Why this answers the question (2-4 bullets).
- If assumptions were made, add: Assumptions.

SQL style requirements:
- Uppercase SQL keywords.
- Clean indentation and aliases.
- Explicit JOIN keys.
- Deterministic ORDER BY when returning top/bottom rows.

If schema context is provided in the conversation, prioritize it. If critical schema details are missing, ask one focused clarification question.
"""


# --- State ---
class ChatState(TypedDict):
    question: Required[str]
    schema_context: NotRequired[str]
    response: NotRequired[str]


# --- OpenAI tool schema ---
RETRIEVE_CONTEXT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_context",
        "description": (
            "Look up Databricks table, column, rule, and example context"
            " relevant to a question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                # "question": {
                #     "type": "string",
                #     "description": "The user's analytics question or the query intent.",
                # },
                "table": {
                    "type": ["string"],
                    "description": "table name to narrow retrieval based on user's analytics question or the query intent.",
                },
            },
            "required": ["table"],
            "additionalProperties": False,
        },
    },
}


# --- Client factory ---
def _build_client() -> OpenAI:
    """Create and return an authenticated OpenAI client.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set in the environment.
    """
    api_key = os.getenv(_OPENAI_API_KEY_VAR)
    if not api_key:
        raise RuntimeError(
            f"{_OPENAI_API_KEY_VAR} is not set. "
            "Add it to your .env file or export it in your shell."
        )
    log.debug("OpenAI client initialised (model=%s).", _MODEL)
    return OpenAI(api_key=api_key)


client = _build_client()


# --- Helper ---
def _serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    """Convert an OpenAI ToolCall object to a plain dict for message history."""
    fn = tool_call.function
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {"name": fn.name, "arguments": fn.arguments},
    }


# --- LangGraph node ---
def _draft_response_node(state: ChatState) -> dict[str, str]:
    question = state["question"]
    schema_context = state.get("schema_context", "")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(schema_context)},
        {"role": "user", "content": question},
    ]

    while True:
        completion = client.chat.completions.create(
            model=_MODEL,
            messages=cast(Any, messages),
            tools=cast(Any, [RETRIEVE_CONTEXT_TOOL]),
            tool_choice="auto",
        )
        message = completion.choices[0].message
        tool_calls = cast(list[Any], message.tool_calls or [])

        if not tool_calls:
            return {"response": message.content or ""}

        # Append assistant turn with all tool-call requests
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [_serialize_tool_call(tc) for tc in tool_calls],
            }
        )

        # Dispatch each tool call and append results
        for tc in tool_calls:
            arguments = json.loads(cast(Any, tc).function.arguments or "{}")
            tool_result = retrieve_context.invoke(
                {
                    "table": arguments["table"],
                }
            )
            print(f"\n[Tool call: retrieve_context with args {arguments}]\nResult:\n{tool_result}\n")
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
            )


# --- Agent builder ---
def build_agent(schema_context: str):
    """Compile a LangGraph agent with schema_context baked into every node call."""

    def draft_response_node(state: ChatState) -> dict[str, str]:
        return _draft_response_node({**state, "schema_context": schema_context})

    graph = StateGraph(ChatState)
    graph.add_node("draft_response", draft_response_node)
    graph.add_edge(START, "draft_response")
    graph.add_edge("draft_response", END)
    return graph.compile(checkpointer=InMemorySaver(), store=InMemoryStore())


# --- CLI entrypoint ---
def chat() -> None:
    schema_context = get_schema_text()
    agent = build_agent(schema_context)

    print("Welcome to DataPilot! Ask me anything about your Databricks data. Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            result = agent.invoke(
                {"question": question},
                {"configurable": {"thread_id": uuid4()}},
            )
        except Exception as exc:
            log.error("Agent invocation failed: %s", exc)
            print(f"ERROR: {exc}")
            continue

        print(f"AI Response:\n{result.get('response', '')}")


if __name__ == "__main__":
    chat()
