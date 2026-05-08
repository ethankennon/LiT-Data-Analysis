"""
Extract thermal images from capture_database.db as PNG files with timestamps.

Each thermal_imgs row already stores a PNG bitstream. This script joins thermal_imgs
with the events table on matching IDs to retrieve each frame's timestamp, then converts
the boot-relative nanosecond timestamps to wall-clock UTC using the collection's
dateTime anchor from collection_details.

Output structure:
    <output_dir>/
        collection_<id>_<name>/
            frame_<event_id>_<iso_timestamp>.png
        manifest.csv   (all frames with metadata)
"""

import argparse
import csv
import datetime
import sqlite3
from pathlib import Path


def safe_dirname(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()


def extract_images(db_path: str, output_dir: str, collection_id: int | None = None) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Load collection anchors: {collection_id: (name, dateTime_ms)}
    cursor.execute("SELECT id, name, dateTime FROM collection_details")
    collections = {row["id"]: (row["name"], row["dateTime"]) for row in cursor.fetchall()}

    # For each collection find its first THERMAL_IMG event timestamp (boot ns anchor)
    cursor.execute(
        "SELECT collectionId, MIN(timestamp) AS first_ts "
        "FROM events WHERE type = 'THERMAL_IMG' GROUP BY collectionId"
    )
    first_ts = {row["collectionId"]: row["first_ts"] for row in cursor.fetchall()}

    # Build main query joining events and thermal_imgs on id
    collection_filter = ""
    params: list = []
    if collection_id is not None:
        collection_filter = " AND e.collectionId = ?"
        params.append(collection_id)

    cursor.execute(
        "SELECT e.id, e.collectionId, e.timestamp, t.image, t.minTemp, t.maxTemp "
        "FROM events e "
        "JOIN thermal_imgs t ON t.id = e.id "
        f"WHERE e.type = 'THERMAL_IMG'{collection_filter} "
        "ORDER BY e.timestamp ASC",
        params,
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No matching frames found.")
        return

    manifest_rows = []
    prev_collection_id = None
    frame_dir = None

    for row in rows:
        event_id = row["id"]
        coll_id = row["collectionId"]
        boot_ns = row["timestamp"]
        image_bytes = row["image"]
        min_temp = row["minTemp"]
        max_temp = row["maxTemp"]

        coll_name, date_time_ms = collections.get(coll_id, ("unknown", None))

        # Compute wall-clock UTC timestamp
        if date_time_ms is not None:
            anchor_boot_ns = first_ts.get(coll_id, boot_ns)
            offset_ms = (boot_ns - anchor_boot_ns) / 1_000_000
            wall_ms = date_time_ms + offset_ms
            wall_dt = datetime.datetime.fromtimestamp(wall_ms / 1000, tz=datetime.timezone.utc)
            iso_ts = wall_dt.strftime("%Y%m%dT%H%M%S_%f")[:-3]  # trim to ms
        else:
            # Fall back to raw boot nanoseconds when no wall-clock anchor exists
            iso_ts = f"boot_{boot_ns}ns"
            wall_dt = None

        # Trim temperature values to 2 decimal places for manifest readability
        min_temp = round(min_temp, 2) if min_temp is not None else None
        max_temp = round(max_temp, 2) if max_temp is not None else None
        trimmed_temps = f"{min_temp}C-{max_temp}C" if min_temp is not None and max_temp is not None else "No-Range"

        # One subdirectory per collection
        if coll_id != prev_collection_id:
            dir_name = f"collection_{coll_id}_{safe_dirname(coll_name)}"
            frame_dir = output / dir_name
            frame_dir.mkdir(exist_ok=True)
            prev_collection_id = coll_id

        filename = f"frame_{iso_ts}_{trimmed_temps}.png"
        filepath = frame_dir / filename
        filepath.write_bytes(image_bytes)

        manifest_rows.append(
            {
                "event_id": event_id,
                "collection_id": coll_id,
                "collection_name": coll_name,
                "boot_timestamp_ns": boot_ns,
                "wall_clock_utc": wall_dt.isoformat() if wall_dt else "",
                "min_temp": min_temp,
                "max_temp": max_temp,
                "file": str(filepath.relative_to(output)),
            }
        )

    # Write manifest CSV
    manifest_path = output / "manifest.csv"
    fieldnames = ["event_id", "collection_id", "collection_name",
                  "boot_timestamp_ns", "wall_clock_utc", "min_temp", "max_temp", "file"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Extracted {len(manifest_rows)} frames to: {output}")
    print(f"Manifest written to: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract thermal PNG images from capture_database.db")
    parser.add_argument("db", help="Path to capture_database.db")
    parser.add_argument("output", help="Output directory for PNG files and manifest")
    parser.add_argument(
        "--collection", type=int, default=None,
        help="Extract only frames from this collection ID (default: all collections)"
    )
    args = parser.parse_args()

    extract_images(args.db, args.output, args.collection)


if __name__ == "__main__":
    main()
