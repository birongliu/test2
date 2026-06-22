"""
knowledge.py — Knowledge layer: ingestion + vector-DB lookup
============================================================
Owns everything about the Qdrant knowledge base:
  - Clients (Qdrant + Voyage embeddings)
  - embed()                  : cached embedding
  - get_identity()           : cached catalog/schema/table inventory
  - retrieve_context()       : two-stage retrieval + relationship expansion
                               + business rules + examples
  - date_context()           : Databricks date hints
  - save_example()           : flywheel persistence (human-approved)

Point types in the collection:
  table | column | example | business_rule | relationship

The agent (core.py) imports from here. Ingestion scripts (ingest_table.py,
sync_rules.py) also share these clients/helpers.

Env:
  QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION, VOYAGE_API_KEY,
  EMBED_MODEL (default voyage-3)
"""

import logging
import os
import re
import uuid
from collections import defaultdict
from functools import lru_cache

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, MatchValue,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("knowledge")

# ─────────────────────────────────────────
# Config + clients
# ─────────────────────────────────────────
QDRANT_URL     = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION     = os.environ["QDRANT_COLLECTION"]
EMBED_MODEL    = os.environ.get("EMBED_MODEL", "voyage-3")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# ─────────────────────────────────────────
# Schema identity — resolved ONCE, cached
# ─────────────────────────────────────────
_IDENTITY_CACHE: dict | None = None


def get_identity(refresh: bool = False) -> dict:
    """Resolve catalog/schema/tables from Qdrant once and cache it."""
    global _IDENTITY_CACHE
    if _IDENTITY_CACHE is not None and not refresh:
        return _IDENTITY_CACHE

    result, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="type", match=MatchValue(value="table"))
        ]),
        limit=10,
        with_payload=["catalog", "db_schema", "table", "text"],
        with_vectors=False,
    )
    if not result:
        raise RuntimeError(f"Collection '{COLLECTION}' has no tables. Run ingestion first.")

    catalogs = sorted({p.payload.get("catalog", "")   for p in result if p.payload} - {""})
    schemas  = sorted({p.payload.get("db_schema", "") for p in result if p.payload} - {""})

    table_map: dict[str, dict] = defaultdict(dict)
    for p in result:
        if not p.payload: continue

        cat = p.payload.get("catalog", "")
        sch = p.payload.get("db_schema", "")
        tbl = p.payload["table"]
        table_map[tbl] = {
            "catalog": cat, "db_schema": sch, "fqn": f"{cat}.{sch}.{tbl}", "description": p.payload.get("text", "")
        }

    inventory: dict[str, list[str]] = defaultdict(list)
    for p in result:
        if not p.payload: continue

        cat = p.payload.get("catalog", "")
        sch = p.payload.get("db_schema", "")
        inventory[f"{cat}.{sch}"].append(p.payload["table"])
    inventory = {k: sorted(v) for k, v in sorted(inventory.items())}

    _IDENTITY_CACHE = {
        "catalogs":  catalogs,
        "schemas":   schemas,
        "table_map": dict(table_map),
        "inventory": inventory,
        "multi":     len(inventory) > 1,
    }
    log.info("Identity cached: %d schema(s), %d distinct table names",
             len(inventory), len(table_map))
    return _IDENTITY_CACHE


def get_schema_text(refresh: bool = False) -> str:
    """Return an LLM-friendly schema summary text block."""
    identity = get_identity(refresh=refresh)

    lines: list[str] = []
    lines.append("database_context:")
    lines.append(f"  mode: {'multi-catalog' if identity.get('multi') else 'single-catalog'}")

    lines.append("  catalogs:")
    catalogs = identity.get("catalogs", [])
    if catalogs:
        for catalog in catalogs:
            lines.append(f"    - {catalog}")
    else:
        lines.append("    - <none>")

    lines.append("  schemas:")
    schemas = identity.get("schemas", [])
    if schemas:
        for schema in schemas:
            lines.append(f"    - {schema}")
    else:
        lines.append("    - <none>")

    lines.append("  inventory:")
    inventory = identity.get("inventory", {})
    if inventory:
        for namespace, tables in inventory.items():
            lines.append(f"    {namespace}:")
            if tables:
                for table in tables:
                    lines.append(f"      - {table}")
            else:
                lines.append("      - <none>")
    else:
        lines.append("    <none>: []")

    lines.append("  tables:")
    table_map = identity.get("table_map", {})
    if table_map:
        for table_name, meta in sorted(table_map.items()):
            lines.append(f"    - name: {table_name}")
            lines.append(f"      fqn: {meta.get('fqn', '')}")
            lines.append(f"      catalog: {meta.get('catalog', '')}")
            lines.append(f"      schema: {meta.get('db_schema', '')}")
            description = (meta.get("description") or "").strip().replace("\n", " ")
            lines.append(f"      description: {description}")
    else:
        lines.append("    - name: <none>")

    return "\n".join(lines)

print("Knowledge layer loaded. Call get_identity() to cache schema info from Qdrant.")
print(get_identity())
print(get_schema_text())