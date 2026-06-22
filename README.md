# DataPilot: Databricks SQL Copilot

A LangGraph-based AI copilot for writing correct, efficient SQL on Databricks. Built with OpenAI, LangChain, and Qdrant.

## Features

✨ **Smart SQL Generation**

- Schema-aware SQL generation with full table and column context
- Read-only safety (no INSERT/UPDATE/DELETE/MERGE/DDL allowed)
- Semantic retrieval of relevant tables, columns, and examples
- Business rule enforcement

🔍 **Vector-Powered Retrieval**

- Qdrant-backed semantic search over your Databricks schema
- Fast lookup of relevant tables, columns, and examples
- Business rule and relationship context

🧭 **Dual Interfaces**

- CLI chat mode for terminal-based interaction
- Streamlit web UI for browser-based queries

## Setup

### Prerequisites

- Python 3.12+
- Databricks workspace with SQL connector access
- OpenAI API key
- Qdrant instance (local or cloud)

### Environment Variables

Create a `.env` file in the project root:

```bash
# Databricks
DATABRICKS_HOST=<your-workspace-host>
DATABRICKS_HTTP_PATH=<your-cluster-http-path>
DATABRICKS_CLIENT_ID=<service-principal-id>
DATABRICKS_CLIENT_SECRET=<service-principal-secret>
DATABRICKS_CATALOG=main
DATABRICKS_SCHEMA=sales

# OpenAI
OPENAI_API_KEY=<your-api-key>
OPENAI_MODEL=gpt-4o-mini

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=<optional-api-key>
QDRANT_COLLECTION=databricks_docs

# Voyage (for embeddings)
VOYAGE_API_KEY=<your-voyage-api-key>
EMBED_MODEL=voyage-3
```

### Installation

1. Install dependencies:

```bash
pip install -e .
```

Or with UV:

```bash
uv sync
```

2. Start Qdrant (if using Docker):

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

## Usage

### Ingest Databricks Tables

Before querying, ingest your Databricks tables into Qdrant:

```bash
python ingestion.py --table orders --catalog main --schema sales
python ingestion.py --table customers --catalog main --schema sales
python ingestion.py --table ingest_flights --catalog workspace --schema flights
```

For force re-ingestion (e.g., after schema changes):

```bash
python ingestion.py --table orders --force
```

### CLI Mode

Start the interactive chat:

```bash
python main.py
```

Example conversation:

```
Welcome to DataPilot! Ask me anything about your Databricks data. Type 'exit' to quit.

Your question: list all tables
AI Response:
...

Your question: how many planes are on ground in ingest_flights
AI Response:
Interpretation: I will count the number of unique aircraft currently on the ground using the ingest_flights table.

SQL:
SELECT COUNT(DISTINCT icao24) AS planes_on_ground
FROM workspace.flights.ingest_flights
WHERE on_ground = TRUE;
...

Your question: exit
Goodbye!
```

### Streamlit Web UI

Start the web app:

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

Features:

- 💬 Chat interface with message history
- 📋 Optional schema browser
- 🔧 Tool call inspection
- 🎨 Syntax-highlighted SQL output

## Architecture

```
┌─────────────────────────────────────────┐
│         User Interface Layer            │
│  ┌──────────────┐    ┌────────────────┐ │
│  │   CLI Chat   │    │   Streamlit    │ │
│  │   (main.py)  │    │  Web App       │ │
│  └──────────────┘    └────────────────┘ │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│      Agent & Tool Layer                 │
│  ┌──────────────────────────────────┐   │
│  │  LangGraph Agent (main.py)       │   │
│  │  - State management              │   │
│  │  - Tool orchestration            │   │
│  │  - OpenAI integration            │   │
│  └──────┬─────────────────────────┬─┘   │
│         │                         │      │
│    ┌────▼────┐          ┌────────▼──┐   │
│    │retrieve │          │ OpenAI    │   │
│    │_context │          │ LLM       │   │
│    └────┬────┘          └──────────┘    │
└─────────┼─────────────────────────────┘
          │
┌─────────▼──────────────────────────────┐
│    Knowledge Layer (knowledge.py)       │
│  ┌──────────────────────────────────┐   │
│  │  Qdrant Vector DB                │   │
│  │  - Table metadata                │   │
│  │  - Column info                   │   │
│  │  - Business rules                │   │
│  │  - Examples                      │   │
│  └────────────────────────────────┘    │
└─────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────┐
│    Databricks Data Layer               │
│  ┌──────────────────────────────────┐   │
│  │  Workspace                       │   │
│  │  - Tables: ingest_flights, etc.  │   │
│  │  - Schema metadata               │   │
│  │  - SQL execution (read-only)     │   │
│  └──────────────────────────────────┘   │
└────────────────────────────────────────┘
```

## Key Design Decisions

### No Assumptions

The agent is explicitly forbidden from assuming missing table names, columns, or joins. It asks clarifying questions instead.

### Semantic Retrieval

Uses vector embeddings to find relevant tables and columns, not just keyword matching.

### Read-Only Enforcement

System prompt and validation prevent any write operations.

### Schema-First

All queries must use columns and tables present in the ingested schema.

## Troubleshooting

### Empty embedding error

```
Error code: 400 - Invalid 'input[0]': input cannot be an empty string.
```

The retriever was called without a table name. The agent now checks for "list tables" requests and answers from schema context instead.

### Table not found

Ensure the table has been ingested:

```bash
python ingestion.py --table my_table
```

### Qdrant connection error

Check that Qdrant is running and reachable at the URL in `.env`:

```bash
docker ps | grep qdrant
```

### OpenAI API errors

- Check that `OPENAI_API_KEY` is set correctly
- Verify rate limits and account balance on https://platform.openai.com

## Development

### Testing a new table

```bash
# Ingest
python ingestion.py --table my_new_table

# Query
python main.py
# Your question: describe my_new_table
```

### Debugging the agent

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the Streamlit sidebar option to show tool calls:

- Check "Show tool calls" in the sidebar
- This displays the retrieve_context calls and their results

## Safety & Governance

✅ **Read-Only**
All SQL is validated as SELECT-only. No modifications allowed.

✅ **Permission-Aware**
Respects Databricks table and column permissions.

✅ **No Hallucination**
Refuses to invent table names or use columns that don't exist in the schema.

✅ **Sensitive Data**
Warns before aggregating or exposing sensitive fields (email, phone, SSN, DOB, etc.).

## License

Proprietary. Internal use only.
