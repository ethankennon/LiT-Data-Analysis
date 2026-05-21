"""
Build a hotspot accumulation image from thermal frames in capture_database.db.

Events are processed in timestamp order per collection:
  - Thermal images before the first SEND serial message are skipped.
  - The 'e'/'d' SEND cycle period is detected from the average interval between
    consecutive sends of the same command.
  - Phase reference is set to the first 'e' SEND (t=0), so sin is 0 at the
    excitation edge, +1 at the midpoint of the 'e' phase, and -1 at the midpoint
    of the 'd' phase.
  - For each thermal image following an 'e' or 'd' SEND, pixel temperatures are
    multiplied by sin(2π·elapsed/period) before being added to the accumulator.
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

    # Detect cycle period from consecutive sends of first_cmd
    first_cmd_times = [
        r["timestamp"] for r in rows
        if r["type"] == "SERIAL_MSG" and r["direction"] == "SEND" and r["message"] == first_cmd
    ]
    if len(first_cmd_times) >= 2:
        intervals = [first_cmd_times[i + 1] - first_cmd_times[i] for i in range(len(first_cmd_times) - 1)]
        period_ns = sum(intervals) / len(intervals)
    else:
        # Only one send of first_cmd — estimate from half-cycle to the opposite command
        other_cmd = "d" if first_cmd == "e" else "e"
        other_times = [
            r["timestamp"] for r in rows
            if r["type"] == "SERIAL_MSG" and r["direction"] == "SEND" and r["message"] == other_cmd
        ]
        if not other_times:
            print("Cannot determine cycle period: only one SEND event found.")
            return
        period_ns = 2.0 * abs(other_times[0] - first_cmd_times[0])

    # Phase reference: first 'e' SEND is t=0 (sin=0 at excitation edge, +1 at mid-'e', -1 at mid-'d')
    e_times = [
        r["timestamp"] for r in rows
        if r["type"] == "SERIAL_MSG" and r["direction"] == "SEND" and r["message"] == "e"
    ]
    phase_ref_ns = e_times[0] if e_times else first_cmd_times[0]

    print(f"First SEND command: {repr(first_cmd)}")
    print(f"Last matching SEND timestamp: {last_matching_send_ts}")
    print(f"Detected period: {period_ns / 1e9:.4f} s  ({len(first_cmd_times)} '{first_cmd}' sends)")
    print(f"Phase reference timestamp: {phase_ref_ns}")

    # Pass 2: accumulate pixel temperatures weighted by sin(2π·elapsed/period)
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

        elapsed_ns = row["timestamp"] - phase_ref_ns
        sine_weight = np.sin(2 * np.pi * elapsed_ns / period_ns)

        accumulator += sine_weight * temps

        if last_send == "e":
            n_add += 1
        else:
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
