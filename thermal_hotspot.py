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
    multiplied by sin(2π·elapsed/period + φ) and accumulated into 360 matrices,
    one per degree of phase offset φ (0°–359°).
  - Processing stops after the last SEND matching the first SEND command seen.

Two images are saved:
  - Hotspot (φ=0°): accumulator[0] normalised to [0, 255].
  - Phase: each pixel shows the phase offset (0°–359°) whose accumulator had the
    highest value at that position. White = 0°, black = 359°.

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

    # Precompute phase offsets for 360 matrices (0° to 359°)
    phase_offsets_rad = np.deg2rad(np.arange(360, dtype=np.float32))

    # Pass 2: accumulate pixel temperatures weighted by sin(2π·elapsed/period + φ)
    # accumulators shape: (360, height, width), float32 to limit memory (~442 MB)
    accumulators: np.ndarray | None = None
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
        gray = np.array(img, dtype=np.float32)  # shape (height, width)

        if accumulators is None:
            accumulators = np.zeros((360, *gray.shape), dtype=np.float32)

        min_t: float = row["minTemp"]
        max_t: float = row["maxTemp"]
        temps = min_t + (gray / 255.0) * (max_t - min_t)  # shape (H, W)

        elapsed_ns = row["timestamp"] - phase_ref_ns
        base_angle = 2 * np.pi * elapsed_ns / period_ns
        sine_weights = np.sin(base_angle + phase_offsets_rad)  # shape (360,)

        # Accumulate each phase matrix; loop avoids a (360, H, W) intermediate array
        for phi_idx in range(360):
            accumulators[phi_idx] += sine_weights[phi_idx] * temps

        if last_send == "e":
            n_add += 1
        else:
            n_sub += 1

    if accumulators is None:
        print("No thermal frames were accumulated.")
        return

    print(f"Frames added: {n_add}  subtracted: {n_sub}")

    base = str(Path(output_prefix).with_suffix(""))

    # --- Hotspot image (φ = 0°) ---
    hotspot = accumulators[0]
    min_val = float(hotspot.min())
    max_val = float(hotspot.max())
    val_range = max_val - min_val

    print(f"Hotspot accumulator range: [{min_val:.4f}, {max_val:.4f}]")

    if val_range == 0:
        normalised = np.full(hotspot.shape, 128, dtype=np.uint8)
    else:
        normalised = ((hotspot - min_val) / val_range * 255).round().astype(np.uint8)

    hotspot_filename = f"{base}_min{min_val:.4f}_max{max_val:.4f}_add{n_add}_sub{n_sub}.png"
    Image.fromarray(normalised, mode="L").save(hotspot_filename)
    print(f"Saved hotspot  → {hotspot_filename}")

    for deg in (90, 180, 270):
        acc = accumulators[deg]
        lo, hi = float(acc.min()), float(acc.max())
        r = hi - lo
        if r == 0:
            px = np.full(acc.shape, 128, dtype=np.uint8)
        else:
            px = ((acc - lo) / r * 255).round().astype(np.uint8)
        fn = f"{base}_phase{deg}deg_min{lo:.4f}_max{hi:.4f}_add{n_add}_sub{n_sub}.png"
        Image.fromarray(px, mode="L").save(fn)
        print(f"Saved {deg:3d}°     → {fn}")

    # --- Phase image: pixel = index of phase matrix with highest accumulator value ---
    # argmax over axis 0 gives the best-phase index (0–359) per pixel
    best_phase = np.argmax(accumulators, axis=0).astype(np.float32)  # shape (H, W)

    # White (255) = 0°, black (0) = 359°
    phase_pixels = np.round((359.0 - best_phase) / 359.0 * 255.0).astype(np.uint8)

    phase_filename = f"{base}_phase_add{n_add}_sub{n_sub}.png"
    Image.fromarray(phase_pixels, mode="L").save(phase_filename)
    print(f"Saved phase    → {phase_filename}")


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
