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
        
def compare_laps(best_lap_df, compare_lap_df):
    """
    Compares two laps frame by frame using position.
    Finds exactly where the comparison lap lost time vs the best lap.

    best_lap_df    — pandas DataFrame of your fastest lap
    compare_lap_df — pandas DataFrame of the lap being analyzed

    Returns a list of time loss zones with position and speed delta.
    """
    import numpy as np

    best    = best_lap_df.reset_index(drop=True)
    compare = compare_lap_df.reset_index(drop=True)

    losses = []

    for i, row in compare.iterrows():
        cx = row.get("pos_x", 0)
        cz = row.get("pos_z", 0)
        c_speed = row.get("speed_kph", 0)

        # Find the closest frame in the best lap by position
        distances = np.sqrt(
            (best["pos_x"] - cx) ** 2 +
            (best["pos_z"] - cz) ** 2
        )
        closest_idx = distances.idxmin()
        b_speed = best.loc[closest_idx, "speed_kph"]

        speed_delta = round(b_speed - c_speed, 2)

        # Only record where comparison lap was slower
        if speed_delta > 5:
            losses.append({
                "frame":       i,
                "pos_x":       round(cx, 2),
                "pos_z":       round(cz, 2),
                "best_speed":  round(b_speed, 2),
                "your_speed":  round(c_speed, 2),
                "speed_delta": speed_delta,
            })

    return losses


def print_comparison(losses, lap_number):
    """Print the lap comparison report."""
    if not losses:
        print(f"  Lap {lap_number} — no significant time loss vs best lap.")
        return

    print(f"\n[Ghost AI] Lap {lap_number} vs Best Lap")
    print("─" * 45)
    print(f"  Frames where you were slower: {len(losses)}")

    # Find the worst 5 moments
    worst = sorted(losses, key=lambda x: x["speed_delta"], reverse=True)[:5]
    print("\n  Biggest losses:")
    for w in worst:
        print(f"    Frame {w['frame']:4} | "
              f"You: {w['your_speed']:.1f} kph | "
              f"Best: {w['best_speed']:.1f} kph | "
              f"Delta: -{w['speed_delta']:.1f} kph")
    print()
