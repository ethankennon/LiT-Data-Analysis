"""
Sample a single pixel across every thermal frame in capture_database.db, convert its
grayscale value to temperature via linear interpolation between each frame's minTemp
and maxTemp, and write the results to CSV.

Usage:
    python3 pixel_temperature.py capture_database.db 320 240 pixel_temps.csv
    python3 pixel_temperature.py capture_database.db 320 240 pixel_temps.csv --collection 3

Pixel coordinates are zero-based: x is the column (0 to width-1),
y is the row (0 to height-1). Image dimensions are 640×480 (width×height).
"""

import argparse
import csv
import io
import sqlite3

from PIL import Image


def sample_pixel(db_path: str, px: int, py: int, output_csv: str, collection_id: int | None = None, threshold: float = 0.2) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

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

    results = []
    for row in rows:
        img = Image.open(io.BytesIO(row["image"])).convert("L")

        if results == [] and (px >= img.width or py >= img.height):
            raise ValueError(
                f"Pixel ({px}, {py}) is out of bounds for image size {img.width}×{img.height}"
            )

        gray = img.getpixel((px, py))
        min_temp = row["minTemp"]
        max_temp = row["maxTemp"]
        temperature = min_temp + (gray / 255.0) * (max_temp - min_temp)

        ts = row["timestamp"]
        prev = results[-1] if results else None
        if prev is None:
            status = "neutral"
        elif ts - prev["boot_timestamp_ns"] > 200_000_000:
            status = "invalid"
        else:
            delta = round(temperature, 4) - prev["temperature_c"]
            if delta > threshold:
                status = "rising"
            elif delta < -threshold:
                status = "falling"
            else:
                status = "neutral"

        results.append({
            "event_id": row["id"],
            "collection_id": row["collectionId"],
            "boot_timestamp_ns": ts,
            "gray_value": gray,
            "min_temp": min_temp,
            "max_temp": max_temp,
            "temperature_c": round(temperature, 4),
            "status": status,
        })

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_id", "collection_id", "boot_timestamp_ns",
                        "gray_value", "min_temp", "max_temp", "temperature_c", "status"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Sampled {len(results)} frames → {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample pixel temperature from thermal frames in capture_database.db"
    )
    parser.add_argument("db", help="Path to capture_database.db")
    parser.add_argument("x", type=int, help="Pixel x coordinate (column, 0-based)")
    parser.add_argument("y", type=int, help="Pixel y coordinate (row, 0-based)")
    parser.add_argument("output", help="Output CSV path")
    parser.add_argument(
        "--collection", type=int, default=None,
        help="Restrict to a single collection ID (default: all collections)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.2,
        help="Temperature change (°C) required to classify a frame as rising/falling (default: 0.2)",
    )
    args = parser.parse_args()

    sample_pixel(args.db, args.x, args.y, args.output, args.collection, args.threshold)


if __name__ == "__main__":
    main()
