
import os
from datetime import datetime
from typing import Any, Mapping

from notion_client import Client

NOTION_TOKEN = os.getenv("NOTION_API_KEY")
DB_ID        = os.getenv("NOTION_DB_ID")

def _client() -> Client:
    if NOTION_TOKEN is None:
        raise RuntimeError("NOTION_API_KEY not set")
    return Client(auth=NOTION_TOKEN)

def _title_prop_name(db_json: Mapping[str, object]) -> str:
    """Return the name of the *title* column for a DB."""
    return next(
        name for name, spec in db_json["properties"].items() if spec["type"] == "title"
    )

def ensure_columns(notion: Client, db_id: str, data: dict[str, Any]) -> None:
    """Add any columns in *data* that don't yet exist in the Notion DB."""
    db_obj = notion.databases.retrieve(database_id=db_id)
    existing = db_obj["properties"].keys()

    additions: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if key in existing:
            continue
        # Heuristic: numbers → Number column, everything else → Rich text
        if isinstance(value, (int, float)):
            additions[key] = {"number": {"format": "number"}}
        else:
            additions[key] = {"rich_text": {}}

    if additions:           # only call the API if something is missing
        notion.databases.update(
            database_id=db_id,
            properties=additions,
        )

def push_summary(summary: dict[str, object]) -> str:
    """
    Create a row in the configured Notion DB and return its URL.

    *Strings* become `rich_text` (or the mandatory *title* field),
    *numbers / bools* become `number` properties.
    
    Auto-creates any missing columns before inserting the data.
    """
    if DB_ID is None:
        raise RuntimeError("NOTION_DB_ID not set")

    notion = _client()
    
    # Auto-create any missing columns first
    ensure_columns(notion, DB_ID, summary)
    
    db_schema = notion.databases.retrieve(database_id=DB_ID)
    title_col = _title_prop_name(db_schema)

    # ---------- build Notion-style property map ---------------------------
    props: dict[str, object] = {}
    for key, val in summary.items():
        if key == "kinfer_file":  # shorten for readability
            val = os.path.basename(str(val))

        if isinstance(val, (int, float)):
            props[key] = {"number": val}
        else:
            props[key] = {"rich_text": [{"text": {"content": str(val)}}]}

    # Title (must be present)
    stamp = datetime.fromisoformat(summary["timestamp"])
    friendly_title = f"{summary['eval_name']}  {stamp:%Y-%m-%d %H:%M:%S}"
    props[title_col] = {
        "title": [{"text": {"content": friendly_title}}]
    }

    page = notion.pages.create(parent={"database_id": DB_ID}, properties=props)
    return page["url"]
