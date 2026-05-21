"""
Build a hotspot accumulation image from thermal frames in capture_database.db.

Events are processed in timestamp order per collection:
  - Thermal images before the first SEND serial message are skipped.
  - The most recent SEND message determines how subsequent thermals are accumulated:
      'e'  → add pixel temperatures to the accumulator
      'd'  → subtract pixel temperatures from the accumulator
  - Any other SEND message still updates the tracked command (pausing accumulation).
  - Processing stops after the last SEND matching the first SEND command seen.

The accumulator is normalised to [0, 255] and saved as a grayscale PNG whose
filename encodes the matrix min/max and frame counts.

Dependencies: Pillow, numpy

Usage:
    python3 thermal_hotspot.py [capture_database.db] [hotspot.png]
    python3 thermal_hotspot.py capture_database.db hotspot.png --collection 3
"""

import argparse
import io
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image


def build_hotspot(db_path: str, output_prefix: str, collection_id: int | None = None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    collection_clause = " AND e.collectionId = ?" if collection_id is not None else ""
    params: list = [collection_id] if collection_id is not None else []

    cur.execute(f"""
        SELECT
            e.id,
            e.type,
            e.collectionId,
            e.timestamp,
            t.image    AS img_blob,
            t.minTemp,
            t.maxTemp,
            sm.direction,
            sm.message
        FROM events e
        LEFT JOIN thermal_imgs t  ON t.id  = e.id AND e.type = 'THERMAL_IMG'
        LEFT JOIN serial_msgs  sm ON sm.id = e.id AND e.type = 'SERIAL_MSG'
        WHERE 1=1 {collection_clause}
        ORDER BY e.collectionId, e.timestamp
    """, params)

    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No events found.")
        return

    # Pass 1: determine first SEND command and the timestamp of the last matching SEND
    first_cmd: str | None = None
    last_matching_send_ts: int | None = None

    for row in rows:
        if row["type"] == "SERIAL_MSG" and row["direction"] == "SEND":
            if first_cmd is None:
                first_cmd = row["message"]
            if row["message"] == first_cmd:
                last_matching_send_ts = row["timestamp"]

    if first_cmd is None:
        print("No SEND serial messages found.")
        return

    print(f"First SEND command: {repr(first_cmd)}")
    print(f"Last matching SEND timestamp: {last_matching_send_ts}")

    # Pass 2: accumulate pixel temperatures
    accumulator: np.ndarray | None = None
    n_add = n_sub = 0
    seen_first_serial = False
    last_send: str | None = None

    for row in rows:
        if row["type"] == "SERIAL_MSG":
            seen_first_serial = True
            if row["direction"] == "SEND":
                last_send = row["message"]
            continue

        if row["type"] != "THERMAL_IMG":
            continue

        if not seen_first_serial:
            continue

        if row["timestamp"] > last_matching_send_ts:
            continue

        if last_send not in ("e", "d"):
            continue

        img = Image.open(io.BytesIO(row["img_blob"])).convert("L")
        gray = np.array(img, dtype=np.float64)  # shape (height, width)

        if accumulator is None:
            accumulator = np.zeros(gray.shape, dtype=np.float64)

        min_t: float = row["minTemp"]
        max_t: float = row["maxTemp"]
        temps = min_t + (gray / 255.0) * (max_t - min_t)

        if last_send == "e":
            accumulator += temps
            n_add += 1
        else:
            accumulator -= temps
            n_sub += 1

    if accumulator is None:
        print("No thermal frames were accumulated.")
        return

    min_val = float(accumulator.min())
    max_val = float(accumulator.max())
    val_range = max_val - min_val

    print(f"Accumulator range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"Frames added: {n_add}  subtracted: {n_sub}")

    if val_range == 0:
        normalised = np.full(accumulator.shape, 128, dtype=np.uint8)
    else:
        normalised = ((accumulator - min_val) / val_range * 255).round().astype(np.uint8)

    out_img = Image.fromarray(normalised, mode="L")

    base = str(Path(output_prefix).with_suffix(""))
    filename = f"{base}_min{min_val:.4f}_max{max_val:.4f}_add{n_add}_sub{n_sub}.png"
    out_img.save(filename)
    print(f"Saved → {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Accumulate thermal frames into a hotspot image."
    )
    parser.add_argument(
        "db", nargs="?", default="capture_database.db",
        help="Path to capture_database.db (default: capture_database.db)",
    )
    parser.add_argument(
        "output", nargs="?", default="hotspot.png",
        help="Output image prefix (default: hotspot.png); stats are appended to the filename",
    )
    parser.add_argument(
        "--collection", type=int, default=None,
        help="Restrict to a single collection ID (default: all collections)",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        parser.error(f"Database not found: {args.db}")

    build_hotspot(args.db, args.output, args.collection)


if __name__ == "__main__":
    main()
