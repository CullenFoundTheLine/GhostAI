# telemetry/session.py
# Captures session info before recording starts
# Track, car, weather, tires — everything Ghost AI needs
# to know what it's storing and compare correctly later

from datetime import datetime

CATEGORIES = ["Formula", "GrandTouring", "Prototype", "TouringCar", "Drift", "Rally", "Sim"]

CARS = {
    "Formula":      ["F1", "F2", "F3", "IndyCar", "Formula E"],
    "GrandTouring": ["GT3", "GT4", "GT500"],
    "Prototype":    ["Hypercar", "LMDh", "LMP2"],
    "TouringCar":   ["TCR", "BTCC", "DTM"],
    "Drift":        ["Formula Drift", "D1GP"],
    "Rally":        ["WRC Rally1", "WRC Rally2"],
    "Sim":          ["GT7", "iRacing", "Assetto Corsa", "F1 25"],
}

TRACKS = [
    "Monaco", "Silverstone", "Spa-Francorchamps", "Monza", "Suzuka",
    "Interlagos", "Nurburgring", "Le Mans", "Daytona", "Laguna Seca",
    "Road Atlanta", "Long Beach", "Irwindale", "Trial Mountain",
    "Deep Forest", "Tokyo Expressway", "Sardegna", "Other"
]

WEATHER = ["Dry", "Wet", "Mixed"]
TIRES   = ["Sport", "Racing", "Rain", "Comfort"]
SESSION = ["Practice", "Qualifying", "Race", "Time Trial", "Drift"]


class Session:
    """
    Holds all the information about one driving session.
    Created before recording starts.
    Attached to every lap stored.
    """

    def __init__(self, driver, track, car, weather, tires, session_type):
        self.driver       = driver
        self.track        = track
        self.car          = car
        self.weather      = weather
        self.tires        = tires
        self.session_type = session_type
        self.started_at   = datetime.now().isoformat()

    def to_dict(self):
        return {
            "driver":       self.driver,
            "track":        self.track,
            "car":          self.car,
            "weather":      self.weather,
            "tires":        self.tires,
            "session_type": self.session_type,
            "started_at":   self.started_at,
        }

    def summary(self):
        print("\n--- SESSION ---")
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}")
        print("---------------\n")


def setup():
    print("\n--- GHOST AI SESSION SETUP ---\n")

    driver = input("  Driver name: ").strip()

    print("\n  Racing category:")
    for i, c in enumerate(CATEGORIES):
        print(f"    {i+1}. {c}")
    cat_pick = int(input("  Pick number: ")) - 1
    category = CATEGORIES[cat_pick]

    print(f"\n  Car type ({category}):")
    car_list = CARS[category]
    for i, c in enumerate(car_list):
        print(f"    {i+1}. {c}")
    car_pick = int(input("  Pick number: ")) - 1
    car = car_list[car_pick]

    print("\n  Track:")
    for i, t in enumerate(TRACKS):
        print(f"    {i+1}. {t}")
    track_pick = int(input("  Pick number: ")) - 1
    track = TRACKS[track_pick]

    print("\n  Weather:")
    for i, w in enumerate(WEATHER):
        print(f"    {i+1}. {w}")
    weather = WEATHER[int(input("  Pick number: ")) - 1]

    print("\n  Tires:")
    for i, t in enumerate(TIRES):
        print(f"    {i+1}. {t}")
    tires = TIRES[int(input("  Pick number: ")) - 1]

    print("\n  Session type:")
    for i, s in enumerate(SESSION):
        print(f"    {i+1}. {s}")
    session_type = SESSION[int(input("  Pick number: ")) - 1]

    session = Session(driver, track, car, weather, tires, session_type)
    session.summary()

    confirm = input("  Start session? (yes / no): ").strip().lower()
    if confirm != "yes":
        print("\n  Session cancelled.\n")
        return None

    return session


if __name__ == "__main__":
    s = setup()
    if s:
        print("Session ready:")
        print(s.to_dict())
