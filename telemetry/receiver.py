# telemetry/receiver.py
# GT7 Live Telemetry Receiver
# Connects to PS5, captures session info first, records to CSV

import csv
import os
from datetime import datetime
from gt_telem import TurismoClient
from telemetry.session import setup
from telemetry.parser import Parser

# Get PS5 IP from environment — never hardcoded
PS_IP = os.getenv("TELEMETRY_IP")

if not PS_IP:
    print("No PS5 IP set.")
    print("Run: export TELEMETRY_IP=10.0.0.133")
    exit()

# Step 1 — get session info before recording
session = setup()
if not session:
    exit()

# Step 2 — create CSV file tagged with session info
os.makedirs("data", exist_ok=True)
filename = f"data/{session.track}_{session.car}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

csvfile = open(filename, 'w', newline='')
writer  = csv.writer(csvfile)

writer.writerow([
    'driver', 'track', 'car', 'weather', 'tires', 'session_type',
    'speed', 'gear', 'throttle', 'brake', 'rpm',
    'lap', 'pos_x', 'pos_y', 'pos_z',
    'tire_fl', 'tire_fr', 'tire_rl', 'tire_rr',
    'timestamp'
])

# Step 3 — create parser with 100ms interval
parser = Parser(interval_ms=100)

frame_count = 0

def handle_data(t):
    global frame_count
    frame_count += 1

    # Only parse every 100ms
    data = parser.parse(t, session)
    if data is None:
        return

    writer.writerow([
        data["driver"], data["track"], data["car"],
        session.weather, session.tires, session.session_type,
        data["speed"], data["gear"], data["throttle"], data["brake"],
        round(t.engine_rpm) if t.engine_rpm else 0,
        data["lap"],
        data["pos_x"], data["pos_y"], data["pos_z"],
        round(t.tire_fl_temp, 1), round(t.tire_fr_temp, 1),
        round(t.tire_rl_temp, 1), round(t.tire_rr_temp, 1),
        data["timestamp"]
    ])
    csvfile.flush()

    # One updating line in terminal
    print(
        f"\r  Lap {data['lap']} | "
        f"{data['speed']:.1f} kph | "
        f"G{data['gear']} | "
        f"T:{int(data['throttle'])} "
        f"B:{int(data['brake'])} | "
        f"Frames: {frame_count}    ",
        end="", flush=True
    )

print(f"\n[Ghost AI] Recording to: {filename}\n")

try:
    client = TurismoClient(ps_ip=PS_IP)
    client.register_callback(handle_data)
    client.run()
except KeyboardInterrupt:
    print(f"\n\n[Ghost AI] Session saved: {filename}")
    print(f"[Ghost AI] Total frames captured: {frame_count}")
finally:
    csvfile.close()
