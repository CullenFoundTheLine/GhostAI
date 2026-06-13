# telemetry/receiver.py
# GT7 Live Telemetry Receiver
# Parser merged in — no separate parser.py needed

import csv
import os
from datetime import datetime
from gt_telem import TurismoClient
from telemetry.session import setup

# ── Parser ─────────────────────────────────────────────

class Parser:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.last_saved  = None

    def should_parse(self):
        if self.last_saved is None:
            return True
        elapsed = (datetime.now() - self.last_saved).total_seconds() * 1000
        return elapsed >= self.interval_ms

    def parse(self, t, session):
        if not self.should_parse():
            return None
        self.last_saved = datetime.now()
        return {
            "driver":    session.driver,
            "track":     session.track,
            "car":       session.car,
            "weather":   session.weather,
            "speed":     getattr(t, "speed_kph", 0) or 0,
            "brake":     getattr(t, "brake", 0) or 0,
            "throttle":  getattr(t, "throttle", 0) or 0,
            "gear":      getattr(t, "current_gear", 0) or 0,
            "lap":       getattr(t, "current_lap", 0) or 0,
            "pos_x":     getattr(t, "position_x", 0) or 0,
            "pos_y":     getattr(t, "position_y", 0) or 0,
            "pos_z":     getattr(t, "position_z", 0) or 0,
            "timestamp": datetime.now().isoformat(),
        }


# ── Receiver ───────────────────────────────────────────

PS_IP = os.getenv("TELEMETRY_IP")

if not PS_IP:
    print("No PS5 IP set.")
    print("Run: export TELEMETRY_IP=10.0.0.133")
    exit()

session = setup()
if not session:
    exit()

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

parser      = Parser(interval_ms=100)
frame_count = 0

def handle_data(t):
    global frame_count
    frame_count += 1
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
    print(f"[Ghost AI] Frames captured: {frame_count}")
finally:
    csvfile.close()
