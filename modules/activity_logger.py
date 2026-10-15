"""Activity logger for recording and retrieving user actions and lifecycle events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import METADATA_DIR

ACTIVITY_LOG_FILE = METADATA_DIR / "activity_log.json"


def log_activity(
    event_type: str,
    name: str,
    description: str,
    status: str = "Completed",
) -> None:
    """Record an activity event to persistent metadata storage."""
    timestamp = datetime.now(timezone.utc).strftime("%b %d, %H:%M")
    entry: dict[str, Any] = {
        "type": event_type,
        "name": name,
        "description": description,
        "status": status,
        "date": timestamp,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
    }
    history = get_recent_activity(limit=50)
    history.insert(0, entry)
    try:
        ACTIVITY_LOG_FILE.write_text(json.dumps(history[:50], indent=2), encoding="utf-8")
    except Exception:
        pass


def get_recent_activity(limit: int = 10) -> list[dict[str, Any]]:
    """Retrieve the latest user and model lifecycle activities."""
    if not ACTIVITY_LOG_FILE.exists():
        return []
    try:
        data = json.loads(ACTIVITY_LOG_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[:limit]
        return []
    except Exception:
        return []


def clear_activity_log() -> None:
    """Clear recorded activities (for testing or reset)."""
    if ACTIVITY_LOG_FILE.exists():
        try:
            ACTIVITY_LOG_FILE.unlink()
        except Exception:
            pass
