# telemetry/parser.py
# Takes raw GT7 frames and returns clean data
# Only saves a frame if enough milliseconds have passed

from datetime import datetime


class Parser:
    """
    Samples GT7 frames at a set interval.
    Default: every 100ms
    """

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
