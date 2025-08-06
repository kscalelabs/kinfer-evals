"""Push results to Notion."""

import mimetypes
import os
import pathlib
from datetime import datetime
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import requests
from notion_client import Client

NOTION_TOKEN = os.getenv("NOTION_API_KEY")
DB_ID = os.getenv("NOTION_DB_ID")
NOTION_VER = "2022-06-28"  # required for the file-upload APIs


def _hdr() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VER,
        "accept": "application/json",
    }


def _upload_file(path: pathlib.Path) -> str:
    """Upload *path* (≤20 MB) to Notion and return its file_upload.id."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    r = requests.post(
        "https://api.notion.com/v1/file_uploads",
        json={"filename": path.name, "content_type": mime},
        headers=_hdr() | {"content-type": "application/json"},
    )
    r.raise_for_status()
    obj = r.json()
    upload_id = obj["id"]

    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, mime)}
        r2 = requests.post(
            f"https://api.notion.com/v1/file_uploads/{upload_id}/send",
            files=files,
            headers=_hdr(),
            timeout=30,
        )
    r2.raise_for_status()
    return upload_id


def _client() -> Client:
    if NOTION_TOKEN is None:
        raise RuntimeError("NOTION_API_KEY not set")
    return Client(auth=NOTION_TOKEN)


def _title_prop_name(db_json: Mapping[str, object]) -> str:
    """Return the name of the *title* column for a DB."""
    return next(name for name, spec in db_json["properties"].items() if spec["type"] == "title")


def ensure_columns(notion: Client, db_id: str, data: dict[str, Any]) -> None:
    """Add any columns in *data* that don't yet exist in the Notion DB."""
    db_obj = notion.databases.retrieve(database_id=db_id)
    existing = db_obj["properties"].keys()

    additions: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if key in existing:
            continue

        # Submitted timestamp
        if key == "submitted":
            additions[key] = {"date": {}}

        elif isinstance(value, (int, float)):
            additions[key] = {"number": {"format": "number"}}
        else:
            additions[key] = {"rich_text": {}}

    if additions:  # only call the API if something is missing
        notion.databases.update(
            database_id=db_id,
            properties=additions,
        )


def push_summary(
    summary: dict[str, object],
    files: Sequence[pathlib.Path] | None = None,  # png / mp4 / …
) -> str:
    """Create a row in the configured Notion DB and return its URL.

    *Strings* become `rich_text` (or the mandatory *title* field),
    *numbers / bools* become `number` properties.

    Auto-creates any missing columns before inserting the data.
    """
    if DB_ID is None:
        raise RuntimeError("NOTION_DB_ID not set")

    notion = _client()

    try:
        stamp = datetime.fromisoformat(summary["timestamp"])
    except ValueError:
        stamp = datetime.strptime(summary["timestamp"], "%Y%m%d-%H%M%S")

    # Correct timezone to PST/PDT
    stamp_ca = stamp.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    summary_with_submitted = {**summary, "submitted": stamp_ca.isoformat()}

    # Auto-create any missing columns first
    ensure_columns(notion, DB_ID, summary_with_submitted)

    db_schema = notion.databases.retrieve(database_id=DB_ID)
    title_col = _title_prop_name(db_schema)

    props: dict[str, object] = {}
    for key, val in summary.items():
        if key == "kinfer_file":  # shorten for readability
            val = os.path.basename(str(val))

        if isinstance(val, (int, float)):
            props[key] = {"number": val}
        else:
            props[key] = {"rich_text": [{"text": {"content": str(val)}}]}

    friendly_title = os.path.basename(str(summary["kinfer_file"]))
    props[title_col] = {"title": [{"text": {"content": friendly_title}}]}

    props["submitted"] = {"date": {"start": stamp_ca.isoformat()}}

    page = notion.pages.create(parent={"database_id": DB_ID}, properties=props)

    if files:
        children: list[dict[str, Any]] = []
        for p in files:
            fid = _upload_file(p)
            ext = p.suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".gif"}:
                block_type = "image"
            elif ext in {".mp4", ".mov", ".mkv", ".webm"}:
                block_type = "video"
            else:
                block_type = "file"

            children.append(
                {
                    "object": "block",
                    "type": block_type,
                    block_type: {
                        "type": "file_upload",
                        "file_upload": {"id": fid},
                        "caption": [{"type": "text", "text": {"content": p.name}}],
                    },
                }
            )
        # append in a single PATCH
        requests.patch(
            f"https://api.notion.com/v1/blocks/{page['id']}/children",
            json={"children": children},
            headers=_hdr() | {"Content-Type": "application/json"},
        ).raise_for_status()

    return page["url"]
