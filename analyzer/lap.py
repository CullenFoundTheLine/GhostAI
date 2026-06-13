# analyzer/lap.py
# Summarizes one lap by finding all braking zones
# and calculating time lost at each turn

from analyzer.braking import detect_braking_zones


def summarize_lap(frames, session):
    """
    Takes all frames from one lap.
    Finds every braking zone.
    Returns a lap summary with time lost per turn.

    frames  — list of parsed frame dicts for one lap
    session — Session object
    """
    if not frames:
        return None

    # Find all braking zones in this lap
    zones = detect_braking_zones(frames, session)

    if not zones:
        return {
            "lap":          frames[0].get("lap", 0),
            "driver":       session.driver,
            "track":        session.track,
            "car":          session.car,
            "total_frames": len(frames),
            "turns_found":  0,
            "total_time_lost": 0.0,
            "zones":        []
        }

    total_time_lost = sum(z["seconds_lost"] for z in zones)

    summary = {
        "lap":             frames[0].get("lap", 0),
        "driver":          session.driver,
        "track":           session.track,
        "car":             session.car,
        "total_frames":    len(frames),
        "turns_found":     len(zones),
        "total_time_lost": round(total_time_lost, 2),
        "zones":           zones
    }

    return summary


def split_laps(frames):
    """
    Takes all frames from a full session.
    Splits them into individual laps by lap number.
    Returns a dict: lap_number -> list of frames
    """
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
    """Print a readable lap summary."""
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
