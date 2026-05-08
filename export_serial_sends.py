"""
Export serial SEND events to CSV with timing relative to collection start
and delay to the next RECEIVE (only when it arrives before the next SEND).
"""

import sqlite3
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def export_serial_sends(db_path: str, output_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
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
        WHERE cd.dateTime IS NOT NULL
        ORDER BY e.collectionId, e.timestamp
    """)
    rows = list(cur.fetchall())
    conn.close()

    col_events: dict[int, list] = defaultdict(list)
    for r in rows:
        col_events[r["collection_id"]].append(r)

    fieldnames = [
        "collection_id",
        "collection_name",
        "event_id",
        "time_since_start_s",
        "data_sent",
        "delay_to_next_receive_s",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for col_id, events in sorted(col_events.items()):
            # Index events by position so we can look ahead from each SEND
            for i, ev in enumerate(events):
                if ev["direction"] != "SEND":
                    continue

                time_since_start_s = (ev["boot_ns"] - ev["first_ns"]) / 1e9

                # Walk forward: if the very next event is a RECEIVE, record delay;
                # if it's another SEND (or end of collection), leave delay blank.
                delay_s = ""
                if i + 1 < len(events):
                    nxt = events[i + 1]
                    if nxt["direction"] == "RECEIVE":
                        delay_s = (nxt["boot_ns"] - ev["boot_ns"]) / 1e9

                writer.writerow({
                    "collection_id": col_id,
                    "collection_name": ev["collection_name"],
                    "event_id": ev["event_id"],
                    "time_since_start_s": f"{time_since_start_s:.6f}",
                    "data_sent": ev["message"],
                    "delay_to_next_receive_s": f"{delay_s:.6f}" if delay_s != "" else "",
                })

    with open(output_path) as f:
        n = sum(1 for _ in f) - 1
    print(f"Wrote {n} SEND events to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export serial SEND events to CSV with timing info."
    )
    parser.add_argument("db", nargs="?", default="capture_database.db",
                        help="Path to capture_database.db (default: capture_database.db)")
    parser.add_argument("output", nargs="?", default="serial_sends.csv",
                        help="Output CSV path (default: serial_sends.csv)")
    args = parser.parse_args()

    if not Path(args.db).exists():
        parser.error(f"Database not found: {args.db}")

    export_serial_sends(args.db, args.output)


if __name__ == "__main__":
    main()
