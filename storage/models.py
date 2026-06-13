# storage/models.py
# Defines the shape of data Ghost AI stores
# Every time loss event has these fields

from datetime import datetime


class TimeLossEvent:
    """
    One corner where the driver lost more than 2 seconds.
    This is what gets stored and learned from.
    """

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
