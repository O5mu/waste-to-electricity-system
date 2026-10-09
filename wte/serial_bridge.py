"""Serial bridge: forwards JSON readings from the Arduino (USB serial) to the API.

The Arduino must print one JSON object per line, e.g.
    {"temperature": 101.2, "pressure": 55.4, "voltage": 13.6}

Run with:  python -m wte.serial_bridge --port COM3
"""

import argparse
import json
import time

import requests
import serial

from wte.config import API_URL


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward Arduino sensor data to the WTE API.")
    parser.add_argument("--port", default="COM3", help="Serial port of the Arduino (e.g. COM3, /dev/ttyACM0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate set in the Arduino sketch")
    parser.add_argument("--api-url", default=API_URL, help="Backend base URL")
    args = parser.parse_args()

    endpoint = f"{args.api_url}/update-reading"
    print(f"Connecting to Arduino on {args.port} @ {args.baud} baud...")

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser, requests.Session() as session:
            time.sleep(2)  # the Arduino resets when the port opens
            print(f"Bridge active -> {endpoint}  (Ctrl+C to stop)")

            while True:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    reading = json.loads(line)
                    session.post(endpoint, json=reading, timeout=2).raise_for_status()
                    print(f"[FORWARDED] {line}")
                except json.JSONDecodeError:
                    print(f"[SKIPPED] Non-JSON serial output: {line}")
                except requests.HTTPError as exc:
                    print(f"[REJECTED] API returned {exc.response.status_code}: {exc.response.text}")
                except requests.RequestException as exc:
                    print(f"[ERROR] Could not reach the API ({exc.__class__.__name__}). Is it running?")

    except serial.SerialException as exc:
        print(f"[SERIAL ERROR] Could not open {args.port}: {exc}")
    except KeyboardInterrupt:
        print("\nBridge stopped.")


if __name__ == "__main__":
    main()
