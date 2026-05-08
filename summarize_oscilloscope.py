import csv
import sys
from pathlib import Path

INPUT_FILE = Path("oscilloscope.csv")
OUTPUT_FILE = Path("oscilloscope_transitions.csv")


def main():
    # --- pass 1: find min/max Volt ---
    volt_min = float("inf")
    volt_max = float("-inf")

    with INPUT_FILE.open() as f:
        reader = csv.reader(f)
        next(reader)  # skip metadata row (x-axis,1)
        next(reader)  # skip header row (second,Volt)
        for row in reader:
            v = float(row[1])
            if v < volt_min:
                volt_min = v
            if v > volt_max:
                volt_max = v

    threshold = (volt_min + volt_max) / 2
    print(f"Volt min: {volt_min}, max: {volt_max}, threshold: {threshold}")

    # --- pass 2: detect threshold crossings ---
    transitions = []

    with INPUT_FILE.open() as f:
        reader = csv.reader(f)
        next(reader)  # skip metadata row
        next(reader)  # skip header row

        prev_high = None  # True if previous sample was above threshold

        for row in reader:
            second = row[0]
            volt = float(row[1])
            is_high = volt > threshold

            if prev_high is None:
                prev_high = is_high
                continue

            if is_high != prev_high:
                label = "off" if is_high else "on"  # low→high = off, high→low = on
                transitions.append((second, volt, label))

            prev_high = is_high

    print(f"Found {len(transitions)} transitions")

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["second", "Volt", "label"])
        writer.writerows(transitions)

    print(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
