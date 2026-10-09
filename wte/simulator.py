"""Sensor simulator: posts realistic readings to the API when no hardware is attached.

The readings come from a synthetic model of the boiler (`wte.plant`) that warms up, follows
hourly waste feeds and drifts, so the 60-minute forecast has something to forecast. Each feed
is also logged to the API, as the operator would.

Run with:  python -m wte.simulator [--interval 2] [--fault-rate 0.1] [--backfill 90]
"""

import argparse
import time
from datetime import datetime, timedelta

import requests

from wte.config import API_URL, PRESSURE_LIMIT_PSI
from wte.plant import PlantProcess


def backfill(plant: PlantProcess, minutes: float, interval: float) -> None:
    """Write `minutes` of simulated history straight into the local database, ending now."""
    from wte import database

    database.init_db()
    start = datetime.now() - timedelta(minutes=minutes)
    last = database.latest_timestamp()
    if last and last >= start.strftime(database.TIME_FORMAT):
        raise SystemExit(f"The database already has readings after {start:%H:%M:%S}; "
                         "backfill only works on an empty or older log (see WTE_DB_PATH).")

    rows, steps = [], int(minutes * 60 / interval)
    for feed in plant.pop_feeds():
        database.insert_feed(**feed, timestamp=start.strftime(database.TIME_FORMAT))
    for step in range(1, steps + 1):
        timestamp = (start + timedelta(seconds=step * interval)).strftime(database.TIME_FORMAT)
        reading = plant.read(interval)
        rows.append((timestamp, reading["temperature"], reading["pressure"], reading["voltage"]))
        for feed in plant.pop_feeds():
            database.insert_readings(rows)
            rows = []
            database.insert_feed(**feed, timestamp=timestamp)
    database.insert_readings(rows)
    print(f"Backfilled {minutes:g} min of history ({steps} readings) into {database.DB_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate WTE plant sensors.")
    parser.add_argument("--api-url", default=API_URL, help="Backend base URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between readings")
    parser.add_argument("--fault-rate", type=float, default=0.1,
                        help="Probability (0-1) of a high-pressure spike; use 0 for stable operation")
    parser.add_argument("--backfill", type=float, default=0, metavar="MINUTES",
                        help="First write this many minutes of past readings directly to the local database, "
                             "so the forecast is available immediately")
    args = parser.parse_args()

    plant = PlantProcess(fault_rate=args.fault_rate)
    if args.backfill:
        backfill(plant, args.backfill, args.interval)

    print(f"WTE sensor simulator -> {args.api_url}  (Ctrl+C to stop)")
    with requests.Session() as session:
        try:
            while True:
                for feed in plant.pop_feeds():
                    try:
                        session.post(f"{args.api_url}/feed", json=feed, timeout=2).raise_for_status()
                        print(f"[FEED] {feed['mass_kg']:.2f} kg loaded; next {feed['next_mass_kg']:.2f} kg "
                              f"in {feed['next_feed_in_min']:.0f} min")
                    except requests.RequestException as exc:
                        print(f"[ERROR] Could not log the feed ({exc.__class__.__name__}).")
                reading = plant.read(args.interval)
                alert = "  ⚠ PRESSURE SPIKE" if reading["pressure"] >= PRESSURE_LIMIT_PSI else ""
                try:
                    session.post(f"{args.api_url}/update-reading", json=reading, timeout=2).raise_for_status()
                    print(f"[SENT] {reading['temperature']:6.2f} °C | "
                          f"{reading['pressure']:6.2f} PSI | {reading['voltage']:5.2f} V{alert}")
                except requests.RequestException as exc:
                    print(f"[ERROR] Could not reach the API ({exc.__class__.__name__}). Is it running?")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
