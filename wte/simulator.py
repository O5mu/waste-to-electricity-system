"""Sensor simulator: posts realistic readings to the API when no hardware is attached.

Run with:  python -m wte.simulator [--interval 2] [--fault-rate 0.1]
"""

import argparse
import random
import time

import requests

from wte.config import API_URL, PRESSURE_LIMIT_PSI


def generate_reading(fault_rate: float) -> dict:
    """One reading from the prototype boiler. `fault_rate` is the chance of a pressure spike."""
    if random.random() < fault_rate:
        pressure = random.uniform(PRESSURE_LIMIT_PSI, PRESSURE_LIMIT_PSI + 6)
    else:
        pressure = random.uniform(50, PRESSURE_LIMIT_PSI - 1)

    return {
        "temperature": round(random.uniform(98.5, 105.0), 2),  # boiling water / steam, °C
        "pressure": round(pressure, 3),                        # PSI
        "voltage": round(random.uniform(13.5, 13.8), 2),       # 12 V DC turbine generator
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate WTE plant sensors.")
    parser.add_argument("--api-url", default=API_URL, help="Backend base URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between readings")
    parser.add_argument("--fault-rate", type=float, default=0.1,
                        help="Probability (0-1) of a high-pressure spike; use 0 for stable operation")
    args = parser.parse_args()

    endpoint = f"{args.api_url}/update-reading"
    print(f"WTE sensor simulator -> {endpoint}  (Ctrl+C to stop)")

    with requests.Session() as session:
        try:
            while True:
                reading = generate_reading(args.fault_rate)
                alert = "  ⚠ PRESSURE SPIKE" if reading["pressure"] >= PRESSURE_LIMIT_PSI else ""
                try:
                    session.post(endpoint, json=reading, timeout=2).raise_for_status()
                    print(f"[SENT] {reading['temperature']:6.2f} °C | "
                          f"{reading['pressure']:6.2f} PSI | {reading['voltage']:5.2f} V{alert}")
                except requests.RequestException as exc:
                    print(f"[ERROR] Could not reach the API ({exc.__class__.__name__}). Is it running?")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
