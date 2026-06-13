# storage/repository.py
# Saves time loss events and averages them by turn and sector
# Only stores events where driver lost more than 2 seconds

import json
import os
from storage.models import TimeLossEvent

# Where events get saved
DATA_DIR  = "data"
EVENTS_FILE = os.path.join(DATA_DIR, "time_loss_events.json")

# Minimum seconds lost to store an event
LOSS_THRESHOLD = 2.0


def save_event(event: TimeLossEvent):
    """
    Save one time loss event to disk.
    Only saves if seconds lost is above threshold.
    """
    if event.seconds_lost < LOSS_THRESHOLD:
        return False

    os.makedirs(DATA_DIR, exist_ok=True)

    # Load existing events
    events = _load_all()

    # Add new event
    events.append(event.to_dict())

    # Save back to disk
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)

    return True


def _load_all():
    """Load all stored events from disk."""
    if not os.path.exists(EVENTS_FILE):
        return []
    with open(EVENTS_FILE, "r") as f:
        return json.load(f)


def get_by_turn(track: str, turn: int):
    """
    Get all events for a specific turn on a specific track.
    Repository calls these to build the average.
    """
    events = _load_all()
    return [
        e for e in events
        if e["track"] == track and e["turn"] == turn
    ]


def average_loss_by_turn(track: str):
    """
    For every turn on a track, calculate the average
    seconds lost across all stored laps.
    Returns a dict: turn number → average seconds lost
    """
    events = _load_all()

    # Filter to this track only
    track_events = [e for e in events if e["track"] == track]

    if not track_events:
        return {}

    # Group by turn
    turns = {}
    for e in track_events:
        turn = e["turn"]
        if turn not in turns:
            turns[turn] = []
        turns[turn].append(e["seconds_lost"])

    # Average each turn
    averages = {}
    for turn, losses in turns.items():
        averages[turn] = round(sum(losses) / len(losses), 3)

    return averages


def average_loss_by_sector(track: str):
    """
    Groups turns into 3 sectors and averages time loss per sector.
    Sector 1 = early turns
    Sector 2 = middle turns
    sector 3 = late turns
    """
    by_turn = average_loss_by_turn(track)

    if not by_turn:
        return {}

    turns = sorted(by_turn.keys())
    total = len(turns)
    third = max(1, total // 3)

    sector1 = turns[:third]
    sector2 = turns[third:third*2]
    sector3 = turns[third*2:]

    def avg(turn_list):
        values = [by_turn[t] for t in turn_list if t in by_turn]
        if not values:
            return 0.0
        return round(sum(values) / len(values), 3)

    return {
        "sector_1": avg(sector1),
        "sector_2": avg(sector2),
        "sector_3": avg(sector3),
    }


def print_report(track: str):
    """Print a full time loss report for a track."""
    print(f"\n[Ghost AI] Time Loss Report — {track}")
    print("─" * 40)

    by_turn = average_loss_by_turn(track)
    if not by_turn:
        print("  No events stored for this track yet.")
        return

    for turn, loss in sorted(by_turn.items()):
        bar = "█" * int(loss * 5)
        print(f"  Turn {turn:2} | {loss:.2f}s lost | {bar}")

    print()
    by_sector = average_loss_by_sector(track)
    print("  Sector averages:")
    for sector, loss in by_sector.items():
        print(f"    {sector}: {loss:.2f}s avg lost")

    print()
