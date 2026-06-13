# analyzer/lap.py
# Lap summarizer + braking zone detector — combined
# braking.py merged in — no separate file needed

from storage.repository import TimeLossEvent, save_event

# ── Braking Detection ──────────────────────────────────

BRAKE_THRESHOLD    = 30
MIN_SPEED_LOSS     = 20.0
TIME_LOSS_THRESHOLD = 2.0


def detect_braking_zones(frames, session):
    zones       = []
    in_brake    = False
    zone_start  = None
    zone_start_idx = 0
    turn_number = 0

    for i, frame in enumerate(frames):
        brake = frame.get("brake", 0) or 0
        speed = frame.get("speed", 0) or 0

        if brake > BRAKE_THRESHOLD and not in_brake:
            in_brake       = True
            zone_start     = frame
            zone_start_idx = i

        elif brake <= BRAKE_THRESHOLD and in_brake:
            in_brake    = False
            turn_number += 1

            if zone_start is None:
                continue

            duration_s = (i - zone_start_idx) * 0.1
            speed_loss = zone_start.get("speed", 0) - speed

            if speed_loss < MIN_SPEED_LOSS:
                continue

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


# ── Lap Summarizer ─────────────────────────────────────

def summarize_lap(frames, session):
    if not frames:
        return None

    zones = detect_braking_zones(frames, session)

    if not zones:
        return {
            "lap":             frames[0].get("lap", 0),
            "driver":          session.driver,
            "track":           session.track,
            "car":             session.car,
            "total_frames":    len(frames),
            "turns_found":     0,
            "total_time_lost": 0.0,
            "zones":           []
        }

    total_time_lost = sum(z["seconds_lost"] for z in zones)

    return {
        "lap":             frames[0].get("lap", 0),
        "driver":          session.driver,
        "track":           session.track,
        "car":             session.car,
        "total_frames":    len(frames),
        "turns_found":     len(zones),
        "total_time_lost": round(total_time_lost, 2),
        "zones":           zones
    }


def split_laps(frames):
    laps = {}
    for frame in frames:
        lap = frame.get("lap", 0)
        if lap < 1:
            continue
        if lap not in laps:
            laps[lap] = []
        laps[lap].append(frame)
    return laps


def print_lap_summary(summary):
    if not summary:
        print("  No lap data.")
        return
    print(f"\n[Ghost AI] Lap {summary['lap']} Summary")
    print(f"  Track:       {summary['track']}")
    print(f"  Car:         {summary['car']}")
    print(f"  Turns found: {summary['turns_found']}")
    print(f"  Total time lost: {summary['total_time_lost']}s")
    print()
    for z in summary["zones"]:
        print(f"  Turn {z['turn']:2} | "
              f"Lost: {z['seconds_lost']:.2f}s | "
              f"Entry: {z['speed_entry']:.1f} kph | "
              f"Brake: {z['brake_pressure']:.0f}")
