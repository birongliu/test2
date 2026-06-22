"""
Ingest a single Databricks table into Qdrant
=============================================
Usage:
  python ingest_table.py --table orders
  python ingest_table.py --table customers --catalog main --schema sales
"""

import argparse
from collections import defaultdict
import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone

from databricks import sql as dbsql
from databricks.sdk.credentials_provider import oauth_service_principal
from databricks.sdk.core import Config
from dotenv import load_dotenv
from langchain import tools
from qdrant_client.models import (
    Distance,
    ExtendedPointId,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from qdrant_client.models import Condition

# Shared knowledge-layer clients & helpers (single source of truth)
from knowledge import qdrant, COLLECTION, EMBED_MODEL
from openai import OpenAI
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("ingest_table")

DATABRICKS_HOST  = os.environ["DATABRICKS_HOST"]
DATABRICKS_HTTP  = os.environ["DATABRICKS_HTTP_PATH"]
DATABRICKS_CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID") or os.environ.get("DATABRICKS_CLIENT")
DATABRICKS_CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET") or os.environ.get("DATABRICKS_TOKEN")
VECTOR_SIZE      = 1024

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def embed(text: str) -> list[float]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    result = client.embeddings.create(input=[text], model=EMBED_MODEL)
    return [float(x) for x in result.data[0].embedding]


def checksum(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def ensure_collection(collection: str):
    existing = [c.name for c in qdrant.get_collections().collections]
    if collection not in existing:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        for field in ("type", "table", "checksum"):
            qdrant.create_payload_index(collection, field, PayloadSchemaType.KEYWORD)
        log.info("Created collection: %s", collection)


def existing_checksums_for_table(collection: str, table: str) -> dict[str, str]:
    """Returns {checksum: point_id} for all points belonging to this table."""
    result_map, offset = {}, None
    while True:
        result, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="table", match=MatchValue(value=table))
            ]),
            limit=500,
            offset=offset,
            with_payload=["checksum"],
            with_vectors=False,
        )
        for p in result:
            if not p.payload:
                continue
            cs = p.payload.get("checksum")
            if cs:
                result_map[cs] = str(p.id)
        if offset is None:
            break
    return result_map


def credential_provider():
    if not DATABRICKS_CLIENT_ID:
        raise ValueError("Missing Databricks client ID. Set DATABRICKS_CLIENT_ID.")
    if not DATABRICKS_CLIENT_SECRET:
        raise ValueError("Missing Databricks client secret. Set DATABRICKS_CLIENT_SECRET for OAuth M2M.")

    config = Config(
        host=DATABRICKS_HOST,
        client_id=DATABRICKS_CLIENT_ID,
        client_secret=DATABRICKS_CLIENT_SECRET,
        auth_type="oauth-m2m",
    )


    return oauth_service_principal(config)


get_token_provider = credential_provider
# ─────────────────────────────────────────
# Fetch one table from Databricks
# ─────────────────────────────────────────
def fetch_table(catalog: str, schema: str, table: str) -> dict:
    log.info("Fetching %s.%s.%s from Databricks…", catalog, schema, table)
    with dbsql.connect(
        server_hostname=DATABRICKS_HOST,
        http_path=DATABRICKS_HTTP,
        credentials_provider=get_token_provider
    ) as conn:
        with conn.cursor() as cursor:
            print(f"""
                SELECT comment
                FROM {catalog}.information_schema.tables
                WHERE table_schema = '{schema}' AND table_name = '{table}'
            """)
            # Table comment
            cursor.execute(f"""
                SELECT comment
                FROM {catalog}.information_schema.tables
                WHERE table_schema = '{schema}' AND table_name='{table}'
            """)
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Table '{table}' not found in {catalog}.{schema}")
            table_comment = row[0] or ""

            # Column comments + types
            cursor.execute(f"""
                SELECT column_name, data_type, comment
                FROM {catalog}.information_schema.columns
                WHERE table_schema = '{schema}'
                  AND table_name   = '{table}'
                ORDER BY ordinal_position
            """)
            columns = [
                {"name": col, "dtype": dtype, "description": comment or ""}
                for col, dtype, comment in cursor.fetchall()
            ]

            # Distinct values for low-cardinality string columns
            for col_info in columns:
                if col_info["dtype"].upper() in ("STRING", "VARCHAR", "CHAR", "BOOLEAN"):
                    try:
                        cursor.execute(f"""
                            SELECT DISTINCT {col_info['name']}
                            FROM {catalog}.{schema}.{table}
                            WHERE {col_info['name']} IS NOT NULL
                            LIMIT 21
                        """)
                        vals = [str(r[0]) for r in cursor.fetchall()]
                        if len(vals) <= 20:
                            col_info["distinct_values"] = vals
                        else:
                            col_info["distinct_values"] = []
                    except Exception:
                        col_info["distinct_values"] = []
                else:
                    col_info["distinct_values"] = []

    log.info("Fetched %d columns", len(columns))
    return {"description": table_comment, "columns": columns}


# ─────────────────────────────────────────
# Build raw points
# ─────────────────────────────────────────
def build_points(table: str, table_info: dict,
                 catalog: str, db_schema: str) -> list[dict]:
    raw = []

    # Table-level point
    t_text = (
        f"{table}: {table_info['description']}"
        if table_info["description"]
        else table
    )
    raw.append({
        "text": t_text,
        "payload": {
            "type":      "table",
            "table":     table,
            "catalog":   catalog,
            "db_schema": db_schema,
            "fqn":      f"{catalog}.{db_schema}.{table}",
            "text":      t_text,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        },
    })

    # Column-level points
    for col in table_info["columns"]:
        parts = [f"{table}.{col['name']} ({col['dtype']})"]
        if col["description"]:
            parts.append(col["description"])
        if col.get("distinct_values"):
            parts.append(f"Valid values: {', '.join(col['distinct_values'])}")
        c_text = ": ".join(parts)
        raw.append({
            "text": c_text,
            "payload": {
                "type":      "column",
                "table":     table,
                "catalog":   catalog,
                "db_schema": db_schema,
                "column":    col["name"],
                "dtype":     col["dtype"],
                "fqn":      f"{catalog}.{db_schema}.{table}",
                "text":      c_text,
                "synced_at": datetime.now(timezone.utc).isoformat(),
            },
        })

    return raw


# ─────────────────────────────────────────
# Upsert (incremental)
# ─────────────────────────────────────────
def upsert_table(collection: str, table: str, raw_points: list[dict],
                 force: bool | None = False) -> None:
    existing = existing_checksums_for_table(collection, table)

    # force=True clears all existing points for this table first, so payload
    # changes (e.g. added catalog/db_schema) are guaranteed to be re-written
    # even when the embedded text is unchanged.
    if force and existing:
        qdrant.delete(collection_name=collection,
                      points_selector=PointIdsList(points=list(existing.values())))
        log.info("Force: deleted %d existing points for '%s'", len(existing), table)
        existing = {}

    to_embed = [r for r in raw_points if checksum(r["text"]) not in existing]
    log.info("%d unchanged, %d to embed", len(raw_points) - len(to_embed), len(to_embed))

    if not to_embed:
        log.info("Nothing changed — skipping.")
        return

    points = []
    for r in to_embed:
        log.info("Embedding: %s", r["text"][:80])
        vec = embed(r["text"])
        cs  = checksum(r["text"])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={**r["payload"], "checksum": cs},
        ))

    qdrant.upsert(collection_name=collection, points=points)
    log.info("Upserted %d points for table '%s'", len(points), table)

    # Remove stale points (e.g. column was dropped)
    current_cs = {checksum(r["text"]) for r in raw_points}
    stale_ids: list[ExtendedPointId] = [pid for cs, pid in existing.items() if cs not in current_cs]
    if stale_ids:
        qdrant.delete(collection_name=collection,
                      points_selector=PointIdsList(points=stale_ids))
        log.info("Deleted %d stale points", len(stale_ids))


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def ingest_table(
    table: str,
    catalog: str | None = None,
    schema: str | None = None,
    force: bool | None = False,
):
    catalog    = catalog or os.environ.get("DATABRICKS_CATALOG", "main")
    schema     = schema  or os.environ.get("DATABRICKS_SCHEMA",  "sales")
    # Single shared collection (from QDRANT_COLLECTION) so ingestion and the
    # agent always read/write the same place. catalog/schema are stored in each
    # point's payload, supporting multiple layers in one collection.
    collection = COLLECTION

    ensure_collection(collection)
    table_info = fetch_table(catalog, schema, table)
    raw_points = build_points(table, table_info, catalog, schema)
    upsert_table(collection, table, raw_points, force=force)

    log.info("Done. Collection: %s | Table: %s.%s.%s",
             collection, catalog, schema, table)



@tools.tool()
def retrieve_context(table: str) -> str:
    """Return table and column context for a formatted table input.

    The retriever searches the Qdrant knowledge base for matching tables and
    columns, then assembles the most relevant schema context for the supplied
    table string.

    Returns a formatted text block containing the matched table metadata,
    related columns, and any supporting rules or examples.
    """

    table = table.strip()
    if not table:
        raise ValueError("retrieve_context requires a non-empty table name.")

    qvec = list(embed(table))
   
    table_filter = Filter(
        must=[FieldCondition(key="type", match=MatchValue(value="table"))],
        should=[
            FieldCondition(key="table", match=MatchValue(value=table)),
            FieldCondition(key="fqn", match=MatchValue(value=table)),
        ],
    )
    # if table_hint:
    #     table_filter.append(FieldCondition(key="table", match=MatchValue(value=table_hint)))

    table_hits = qdrant.query_points(
        collection_name=COLLECTION, 
        query=qvec,
        score_threshold=0.70,
        query_filter=table_filter,
        limit=4,
    )
 
    # Identify matched tables by their FULL identity (catalog, schema, table),
    # so silver.orders and gold.orders are kept distinct.
    matched = []
    for h in table_hits.points:
        if not h.payload:
            continue
        matched.append({
            "catalog":   h.payload.get("catalog", ""),
            "db_schema": h.payload.get("db_schema", ""),
            "table":     h.payload["table"],
            "text":      h.payload["text"],
        })
 
    # FIX (weakness 1): expand the set by following join relationships so a
    # needed join table is present even if it didn't rank by similarity.
    # matched, rel_payloads = _expand_via_relationships(matched)
 
    table_names = list({m["table"] for m in matched})
 
    col_hits = qdrant.query_points(
        collection_name=COLLECTION, 
        query=qvec,
        query_filter=Filter(must=[
            FieldCondition(key="type",  match=MatchValue(value="column")),
            FieldCondition(key="table", match=MatchAny(any=table_names)),
        ]),
        limit=40,   # higher: more tables now in scope
    ).points if table_names else []
 
    example_hits = qdrant.query_points(
        collection_name=COLLECTION, query=qvec,
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="example"))]),
        limit=3,
    )
 
    rule_hits = qdrant.query_points(
        collection_name=COLLECTION, query=qvec,
        query_filter=Filter(must=[
            FieldCondition(key="type", match=MatchValue(value="business_rule"))
        ]),
        limit=6,
    )
 
    # Scope columns by (catalog, schema, table) — NOT by table name alone.
    def key(cat, sch, tbl):
        return f"{cat}.{sch}.{tbl}"
 
    cols_by_fqn = defaultdict(list)
    for h in col_hits:
        if not h.payload:
            continue

        k = key(h.payload.get("catalog", ""),
                h.payload.get("db_schema", ""),
                h.payload["table"])
        cols_by_fqn[k].append(f"    - {h.payload['text']}")
 
    schema_lines = []
    for m in matched:
        fqn = f"{m['catalog']}.{m['db_schema']}.{m['table']}"
        tag = "  (added because it joins to a matched table)" if m.get("via_join") else ""
        schema_lines.append(f"  TABLE {fqn}: {m['text']}{tag}")
        k = key(m["catalog"], m["db_schema"], m["table"])
        cols = cols_by_fqn.get(k)
        if not cols and m.get("via_join"):
            # pull this table's columns explicitly so joins have keys to use
            res, _ = qdrant.scroll(
                collection_name=COLLECTION,
                scroll_filter=Filter(must=[
                    FieldCondition(key="type",  match=MatchValue(value="column")),
                    FieldCondition(key="table", match=MatchValue(value=m["table"])),
                ]),
                limit=40, with_payload=True, with_vectors=False,
            )
            cols = [f"    - {p.payload['text']}" for p in res if p.payload
                    if p.payload.get("catalog", "") == m["catalog"]
                    and p.payload.get("db_schema", "") == m["db_schema"]]
        schema_lines.extend(cols or ["    (no columns found)"])
 
    example_lines = [
        f"  Q: {h.payload['question']}\n  SQL:\n{h.payload['sql']}"
        for h in example_hits.points if h.payload
    ]
    rule_lines = [f"  - {h.payload['text']}" for h in rule_hits.points if h.payload]
    # rel_lines  = [f"  - {p['text']}" for p in rel_payloads]
 
    # If matched tables span more than one schema, flag it so the agent
    # asks the analyst which layer they mean instead of guessing.
    distinct_schemas = {f"{m['catalog']}.{m['db_schema']}" for m in matched}
 
    parts = []
    if rule_lines:
        parts.append(
            "<business_rules>\n"
            "These rules are MANDATORY. Apply every relevant one.\n"
            + "\n".join(rule_lines)
            + "\n</business_rules>"
        )
    # if rel_lines:
    #     parts.append(
    #         "<relationships>\n"
    #         "Use these exact join keys when joining the tables below.\n"
    #         + "\n".join(rel_lines)
    #         + "\n</relationships>"
    #    )
    if len(distinct_schemas) > 1:
        parts.append(
            "<schema_ambiguity>\n"
            f"The matched tables exist in multiple schemas: "
            f"{', '.join(sorted(distinct_schemas))}.\n"
            "Do NOT guess. If the user did not specify which schema/layer, ask them "
            "before generating SQL. Use the exact fully-qualified name (shown after "
            "'TABLE') for whichever they choose.\n"
            "</schema_ambiguity>"
        )
    if len(matched) == 0:
        parts.append(
            "<no_matches>\n"
            "No relevant tables found in the schema. Ask the user to clarify their the table name or check for typos, "
            "but be aware it may reference tables/columns that don't exist.\n"
            "</no_matches>"
        )

    parts.append(
        "<schema>\n"
        "Use the fully-qualified name shown after 'TABLE' exactly as written.\n"
        f"{chr(10).join(schema_lines)}\n</schema>"
    )
    if example_lines:
        parts.append("<similar_examples>\n" + "\n\n".join(example_lines) + "\n</similar_examples>")
    return "\n\n".join(parts)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest one Databricks table into Qdrant")
    parser.add_argument("--table",   required=True, help="Table name, e.g. orders")
    parser.add_argument("--catalog", default=None,  help="Databricks catalog (default: env)")
    parser.add_argument("--schema",  default=None,  help="Databricks schema  (default: env)")
    parser.add_argument("--force",   action="store_true",
                        help="Delete existing points for this table and re-ingest "
                             "(use after a payload/schema change)")
    args = parser.parse_args()

    ingest_table(
        table=args.table,
        catalog=args.catalog,
        schema=args.schema,
        force=args.force,
    )