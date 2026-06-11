# ghostai/receiver.py
# GT7 Live Telemetry Receiver
# KEEP THIS FILE AS IS — it works
# Run this FIRST before anything else when GT7 is open
#
# Usage: python receiver.py

import csv
import os
from datetime import datetime
from gt_telem import TurismoClient

os.makedirs("data", exist_ok=True)
filename = f"data/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

csvfile = open(filename, 'w', newline='')
writer = csv.writer(csvfile)

# This is your schema — every column Ghost AI reads
writer.writerow([
    'speed_kph', 'gear', 'throttle', 'brake', 'rpm',
    'lap', 'pos_x', 'pos_y', 'pos_z',
    'tire_fl', 'tire_fr', 'tire_rl', 'tire_rr'
])

def handle_data(t):
    writer.writerow([
        round(t.speed_kph, 2),
        t.current_gear,
        t.throttle,
        t.brake,
        round(t.engine_rpm),
        t.current_lap,
        round(t.position_x, 2),
        round(t.position_y, 2),
        round(t.position_z, 2),
        round(t.tire_fl_temp, 1),
        round(t.tire_fr_temp, 1),
        round(t.tire_rl_temp, 1),
        round(t.tire_rr_temp, 1)
    ])
    print(f"Lap {t.current_lap} | {t.speed_kph:.1f} kph | "
          f"Gear {t.current_gear} | Throttle {t.throttle} | Brake {t.brake}")

print(f"[Ghost AI] Recording to: {filename}")
print(f"[Ghost AI] Connecting to PS5 at 10.0.0.133...")
print(f"[Ghost AI] Press Ctrl+C to stop and save.\n")

try:
    client = TurismoClient(ps_ip="10.0.0.133")
    client.register_callback(handle_data)
    client.run()
except KeyboardInterrupt:
    print(f"\n[Ghost AI] Session saved: {filename}")
finally:
    csvfile.close()
