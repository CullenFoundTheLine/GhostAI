# storage/repository.py
# TimeLossEvent + repository functions — combined
# models.py merged in here — no separate file needed

import json
import os
from datetime import datetime

# ── TimeLossEvent ──────────────────────────────────────

class TimeLossEvent:
    def __init__(self, driver, track, car, weather,
                 lap, turn, seconds_lost,
                 speed_entry, brake_pressure,
                 pos_x, pos_y, pos_z):
        self.driver         = driver
        self.track          = track
        self.car            = car
        self.weather        = weather
        self.lap            = lap
        self.turn           = turn
        self.seconds_lost   = seconds_lost
        self.speed_entry    = speed_entry
        self.brake_pressure = brake_pressure
        self.pos_x          = pos_x
        self.pos_y          = pos_y
        self.pos_z          = pos_z
        self.recorded_at    = datetime.now().isoformat()

    def to_dict(self):
        return {
            "driver":         self.driver,
            "track":          self.track,
            "car":            self.car,
            "weather":        self.weather,
            "lap":            self.lap,
            "turn":           self.turn,
            "seconds_lost":   self.seconds_lost,
            "speed_entry":    self.speed_entry,
            "brake_pressure": self.brake_pressure,
            "pos_x":          self.pos_x,
            "pos_y":          self.pos_y,
            "pos_z":          self.pos_z,
            "recorded_at":    self.recorded_at,
        }

    def summary(self):
        print(f"  Turn {self.turn} | "
              f"Lost: {self.seconds_lost:.2f}s | "
              f"Entry speed: {self.speed_entry:.1f} kph | "
              f"Brake: {self.brake_pressure:.0f}")


# ── Repository ─────────────────────────────────────────

DATA_DIR    = "data"
EVENTS_FILE = os.path.join(DATA_DIR, "time_loss_events.json")
LOSS_THRESHOLD = 2.0


def save_event(event: TimeLossEvent):
    if event.seconds_lost < LOSS_THRESHOLD:
        return False
    os.makedirs(DATA_DIR, exist_ok=True)
    events = _load_all()
    events.append(event.to_dict())
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)
    return True


def _load_all():
    if not os.path.exists(EVENTS_FILE):
        return []
    with open(EVENTS_FILE, "r") as f:
        return json.load(f)


def get_by_turn(track: str, turn: int):
    events = _load_all()
    return [e for e in events if e["track"] == track and e["turn"] == turn]


def average_loss_by_turn(track: str):
    events = _load_all()
    track_events = [e for e in events if e["track"] == track]
    if not track_events:
        return {}
    turns = {}
    for e in track_events:
        turn = e["turn"]
        if turn not in turns:
            turns[turn] = []
        turns[turn].append(e["seconds_lost"])
    return {turn: round(sum(losses) / len(losses), 3)
            for turn, losses in turns.items()}


def average_loss_by_sector(track: str):
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
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "sector_1": avg(sector1),
        "sector_2": avg(sector2),
        "sector_3": avg(sector3),
    }


def print_report(track: str):
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
