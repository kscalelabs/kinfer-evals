"""Push results to Notion."""

import logging
import mimetypes
import os
import pathlib
from datetime import datetime
from typing import Mapping, Sequence, cast
from zoneinfo import ZoneInfo

import requests
from notion_client import Client

logger = logging.getLogger(__name__)

NOTION_TOKEN = os.getenv("NOTION_API_KEY")
DB_ID = os.getenv("NOTION_DB_ID")
NOTION_VER = "2022-06-28"  # required for the file-upload APIs


def _hdr() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VER,
        "accept": "application/json",
    }


def _upload_file(path: pathlib.Path, session: requests.Session) -> str:
    """Upload *path* (≤20 MB) to Notion and return its file_upload.id."""
    logger.info("Uploading file: %s (%.2f MB)", path.name, path.stat().st_size / 1024 / 1024)
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

    r = session.post(
        "https://api.notion.com/v1/file_uploads",
        json={"filename": path.name, "content_type": mime},
        headers=_hdr() | {"content-type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    obj = r.json()
    upload_id = obj["id"]
    logger.info("Got upload ID for %s: %s", path.name, upload_id)

    with open(path, "rb") as fh:
        files = {"file": (path.name, fh, mime)}
        r2 = session.post(
            f"https://api.notion.com/v1/file_uploads/{upload_id}/send",
            files=files,
            headers=_hdr(),
            timeout=60,  # Increased timeout for large files
        )
    r2.raise_for_status()
    logger.info("Successfully uploaded file: %s", path.name)
    return upload_id


def _client() -> Client:
    if NOTION_TOKEN is None:
        raise RuntimeError("NOTION_API_KEY not set")
    return Client(auth=NOTION_TOKEN)


def _title_prop_name(db_json: Mapping[str, object]) -> str:
    """Return the **name** of the column whose type is `"title"`."""
    props = cast(dict[str, object], db_json.get("properties", {}))
    return next(name for name, spec in props.items() if cast(dict[str, object], spec).get("type") == "title")


def ensure_columns(notion: Client, db_id: str, data: dict[str, object]) -> None:
    """Add any columns in *data* that don't yet exist in the Notion DB."""
    db_obj = cast(dict[str, object], notion.databases.retrieve(database_id=db_id))
    existing = cast(dict[str, object], db_obj.get("properties", {})).keys()

    additions: dict[str, dict[str, object]] = {}
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

    ts_raw = str(summary["timestamp"])
    try:
        stamp = datetime.fromisoformat(ts_raw)
    except ValueError:
        stamp = datetime.strptime(ts_raw, "%Y%m%d-%H%M%S")

    # Correct timezone to PST/PDT
    stamp_ca = stamp.replace(tzinfo=ZoneInfo("America/Los_Angeles"))
    summary_with_submitted = {**summary, "submitted": stamp_ca.isoformat()}

    # Auto-create any missing columns first
    ensure_columns(notion, DB_ID, summary_with_submitted)

    db_schema = cast(dict[str, object], notion.databases.retrieve(database_id=DB_ID))
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

    page = cast(dict[str, object], notion.pages.create(parent={"database_id": DB_ID}, properties=props))
    logger.info("Created Notion page: %s", page["url"])

    # Use a session to manage connections properly and ensure cleanup
    session = requests.Session()
    try:
        if files:
            logger.info("Uploading %d files to Notion page", len(files))
            # Put videos first, then all other files
            files_sorted = sorted(
                files, key=lambda p: (p.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}, p.name)
            )

            children: list[dict[str, object]] = []
            for i, p in enumerate(files_sorted, 1):
                try:
                    logger.info("Uploading file %d/%d: %s", i, len(files_sorted), p.name)
                    fid = _upload_file(p, session)
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
                except Exception as e:
                    logger.error("Failed to upload file %s: %s", p.name, e)
                    # Continue with other files instead of failing completely
                    continue

            if children:
                logger.info("Attaching %d uploaded files to Notion page", len(children))
                # append in a single PATCH
                resp = session.patch(
                    f"https://api.notion.com/v1/blocks/{page['id']}/children",
                    json={"children": children},
                    headers=_hdr() | {"Content-Type": "application/json"},
                    timeout=60,
                )
                resp.raise_for_status()
                logger.info("Successfully attached all files to Notion page")
            else:
                logger.warning("No files were successfully uploaded")
    finally:
        # Explicitly close the session to clean up connections
        session.close()
        logger.info("Closed requests session")

    return str(page["url"])
