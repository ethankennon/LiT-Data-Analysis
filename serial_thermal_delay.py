"""
Export serial SEND events to CSV with delay to the first observed thermal response.

For each SEND of 'e', searches forward (same collection) for the first thermal frame
with status 'rising' or 'invalid' and records the time delta.
For each SEND of 'd', searches for the first frame with status 'falling' or 'invalid'.
Other messages leave delay_to_output_observed blank.

delay_to_output_observed is a float (seconds) when a rising/falling match is found,
or the string "invalid" when the first matching frame has status 'invalid'.

Usage:
    python3 serial_thermal_delay.py capture_database.db <x> <y> output.csv
    python3 serial_thermal_delay.py capture_database.db <x> <y> output.csv --threshold 0.5
    python3 serial_thermal_delay.py capture_database.db <x> <y> output.csv --collection 3
"""

import argparse
import csv
import io
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

from PIL import Image


def load_thermal_frames(conn, px: int, py: int, threshold: float, collection_id: int | None = None) -> dict[int, list]:
    """Return thermal frames keyed by collection_id, each sorted by timestamp, with status."""
    cursor = conn.cursor()
    collection_filter = "" if collection_id is None else " AND e.collectionId = ?"
    params = [] if collection_id is None else [collection_id]
    cursor.execute(
        "SELECT e.id, e.collectionId, e.timestamp, t.image, t.minTemp, t.maxTemp "
        "FROM events e "
        "JOIN thermal_imgs t ON t.id = e.id "
        f"WHERE e.type = 'THERMAL_IMG'{collection_filter} "
        "ORDER BY e.collectionId, e.timestamp ASC",
        params,
    )
    rows = cursor.fetchall()

    col_frames: dict[int, list] = defaultdict(list)
    prev_by_col: dict[int, dict] = {}
    bounds_checked = False

    for row in rows:
        col_id = row["collectionId"]
        ts = row["timestamp"]

        img = Image.open(io.BytesIO(row["image"])).convert("L")

        if not bounds_checked:
            if px >= img.width or py >= img.height:
                raise ValueError(
                    f"Pixel ({px}, {py}) is out of bounds for image size {img.width}×{img.height}"
                )
            bounds_checked = True

        gray = img.getpixel((px, py))
        temperature = round(row["minTemp"] + (gray / 255.0) * (row["maxTemp"] - row["minTemp"]), 4)

        prev = prev_by_col.get(col_id)
        if prev is None:
            status = "neutral"
        elif ts - prev["boot_timestamp_ns"] > 200_000_000:
            status = "invalid"
        else:
            delta = temperature - prev["temperature_c"]
            if delta > threshold:
                status = "rising"
            elif delta < -threshold:
                status = "falling"
            else:
                status = "neutral"

        frame = {"boot_timestamp_ns": ts, "temperature_c": temperature, "status": status}
        col_frames[col_id].append(frame)
        prev_by_col[col_id] = frame

    return dict(col_frames)


def find_next_matching_frame(frames: list, after_ts: int, target_statuses: set) -> dict | None:
    """Return the first frame after after_ts whose status is in target_statuses, or None."""
    timestamps = [f["boot_timestamp_ns"] for f in frames]
    start = bisect_right(timestamps, after_ts)
    for i in range(start, len(frames)):
        if frames[i]["status"] in target_statuses:
            return frames[i]
    return None


def export_combined(db_path: str, px: int, py: int, output_path: str, threshold: float = 0.2, collection_id: int | None = None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print(f"Loading thermal frames for pixel ({px}, {py})...")
    col_frames = load_thermal_frames(conn, px, py, threshold, collection_id)

    collection_filter = "" if collection_id is None else " AND e.collectionId = ?"
    serial_params = [] if collection_id is None else [collection_id]

    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            e.id           AS event_id,
            e.collectionId AS collection_id,
            cd.name        AS collection_name,
            e.timestamp    AS boot_ns,
            sm.direction,
            sm.message,
            MIN(e.timestamp) OVER (PARTITION BY e.collectionId) AS first_ns
        FROM events e
        JOIN serial_msgs        sm ON sm.id  = e.id
        JOIN collection_details cd ON cd.id  = e.collectionId
        WHERE cd.dateTime IS NOT NULL{collection_filter}
        ORDER BY e.collectionId, e.timestamp
    """, serial_params)
    serial_rows = list(cur.fetchall())
    conn.close()

    col_events: dict[int, list] = defaultdict(list)
    for r in serial_rows:
        col_events[r["collection_id"]].append(r)

    fieldnames = [
        "collection_id",
        "collection_name",
        "event_id",
        "boot_timestamp_ns",
        "time_since_start_s",
        "data_sent",
        "delay_to_next_receive_s",
        "delay_to_output_observed",
    ]

    TARGET_STATUSES = {
        "e": {"rising", "invalid"},
        "d": {"falling", "invalid"},
    }

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for col_id, events in sorted(col_events.items()):
            frames = col_frames.get(col_id, [])

            for i, ev in enumerate(events):
                if ev["direction"] != "SEND":
                    continue

                time_since_start_s = (ev["boot_ns"] - ev["first_ns"]) / 1e9

                delay_receive = ""
                if i + 1 < len(events) and events[i + 1]["direction"] == "RECEIVE":
                    delay_receive = (events[i + 1]["boot_ns"] - ev["boot_ns"]) / 1e9

                msg = ev["message"]
                delay_to_output = ""
                target = TARGET_STATUSES.get(msg)
                if target is not None and frames:
                    match = find_next_matching_frame(frames, ev["boot_ns"], target)
                    if match is not None:
                        if match["status"] == "invalid":
                            delay_to_output = "invalid"
                        else:
                            delay_to_output = f"{(match['boot_timestamp_ns'] - ev['boot_ns']) / 1e9:.6f}"

                writer.writerow({
                    "collection_id": col_id,
                    "collection_name": ev["collection_name"],
                    "event_id": ev["event_id"],
                    "boot_timestamp_ns": ev["boot_ns"],
                    "time_since_start_s": f"{time_since_start_s:.6f}",
                    "data_sent": msg,
                    "delay_to_next_receive_s": f"{delay_receive:.6f}" if delay_receive != "" else "",
                    "delay_to_output_observed": delay_to_output,
                })

    with open(output_path) as f:
        n = sum(1 for _ in f) - 1
    print(f"Wrote {n} SEND events → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export serial SEND events with delay to observed thermal response."
    )
    parser.add_argument("db", nargs="?", default="capture_database.db",
                        help="Path to capture_database.db (default: capture_database.db)")
    parser.add_argument("x", type=int, help="Pixel x coordinate (column, 0-based)")
    parser.add_argument("y", type=int, help="Pixel y coordinate (row, 0-based)")
    parser.add_argument("output", nargs="?", default="serial_sends.csv",
                        help="Output CSV path (default: serial_sends.csv)")
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Temperature change (°C) required to classify a frame as rising/falling (default: 0.2)",
    )
    parser.add_argument(
        "--collection", type=int, default=None,
        help="Restrict to a single collection ID (default: all collections)",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        parser.error(f"Database not found: {args.db}")

    export_combined(args.db, args.x, args.y, args.output, args.threshold, args.collection)


if __name__ == "__main__":
    main()
