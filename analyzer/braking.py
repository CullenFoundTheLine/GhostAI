# analyzer/braking.py
# Detects braking zones from a list of parsed frames
# Decides if a braking zone counts as a time loss event

from storage.models import TimeLossEvent
from storage.repository import save_event

# Minimum brake pressure to count as braking
BRAKE_THRESHOLD = 30

# Minimum speed loss during a zone to count as a real corner
MIN_SPEED_LOSS = 20.0

# Minimum seconds lost to store the event
TIME_LOSS_THRESHOLD = 2.0


def detect_braking_zones(frames, session):
    """
    Takes a list of parsed frames from one lap.
    Finds every braking zone.
    Calculates time lost at each zone.
    Saves zones where time lost is above threshold.

    frames  — list of dicts from parser.parse()
    session — Session object with driver, track, car info
    """
    zones  = []
    in_brake = False
    zone_start = None
    turn_number = 0

    for i, frame in enumerate(frames):
        brake = frame.get("brake", 0) or 0
        speed = frame.get("speed", 0) or 0

        # Braking started
        if brake > BRAKE_THRESHOLD and not in_brake:
            in_brake   = True
            zone_start = frame
            zone_start_idx = i

        # Braking ended
        elif brake <= BRAKE_THRESHOLD and in_brake:
            in_brake    = False
            turn_number += 1

            if zone_start is None:
                continue

            # Calculate how long the braking zone lasted
            duration_s = (i - zone_start_idx) * 0.1  # 100ms per frame

            # Speed lost during the zone
            speed_loss = zone_start.get("speed", 0) - speed

            # Skip if not a real corner
            if speed_loss < MIN_SPEED_LOSS:
                continue

            # Estimate time lost — longer brake zone = more time lost
            # This is a simple estimate. Will improve with lap comparison later
            seconds_lost = round(duration_s * 0.4, 2)

            zone = {
                "turn":           turn_number,
                "duration_s":     round(duration_s, 2),
                "speed_entry":    zone_start.get("speed", 0),
                "speed_exit":     speed,
                "speed_loss":     round(speed_loss, 1),
                "brake_pressure": zone_start.get("brake", 0),
                "pos_x":          zone_start.get("pos_x", 0),
                "pos_y":          zone_start.get("pos_y", 0),
                "pos_z":          zone_start.get("pos_z", 0),
                "seconds_lost":   seconds_lost,
                "lap":            zone_start.get("lap", 0),
            }
            zones.append(zone)

            # Save if time lost is above threshold
            if seconds_lost >= TIME_LOSS_THRESHOLD:
                event = TimeLossEvent(
                    driver=session.driver,
                    track=session.track,
                    car=session.car,
                    weather=session.weather,
                    lap=zone["lap"],
                    turn=turn_number,
                    seconds_lost=seconds_lost,
                    speed_entry=zone["speed_entry"],
                    brake_pressure=zone["brake_pressure"],
                    pos_x=zone["pos_x"],
                    pos_y=zone["pos_y"],
                    pos_z=zone["pos_z"]
                )
                saved = save_event(event)
                if saved:
                    print(f"  [Braking] Turn {turn_number} — "
                          f"{seconds_lost}s lost — saved")

    return zones
