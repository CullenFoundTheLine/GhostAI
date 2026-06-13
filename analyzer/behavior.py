# analyzer/behavior.py
# Classifies driving style from lap summaries
# Style is context — not the final output
# The final output is always time lost and how to fix it

STYLES = [
    "aggressive-drift",
    "late-braker",
    "smooth-entry",
    "drift-balanced",
    "balanced"
]


def classify_style(lap_summaries):
    """
    Takes a list of lap summaries from lap.py
    Returns the dominant driving style and confidence

    lap_summaries — list of dicts from summarize_lap()
    """
    if not lap_summaries:
        return {"style": "unknown", "confidence": 0}

    # Collect signals across all laps
    total_zones        = 0
    high_brake_zones   = 0
    high_entry_speed   = 0
    high_speed_loss    = 0
    total_time_lost    = 0.0

    for summary in lap_summaries:
        zones = summary.get("zones", [])
        total_zones     += len(zones)
        total_time_lost += summary.get("total_time_lost", 0)

        for z in zones:
            if z["brake_pressure"] > 220:
                high_brake_zones += 1
            if z["speed_entry"] > 160:
                high_entry_speed += 1
            if z["speed_loss"] > 80:
                high_speed_loss  += 1

    if total_zones == 0:
        return {"style": "unknown", "confidence": 0}

    # Calculate ratios
    brake_ratio  = high_brake_zones / total_zones
    entry_ratio  = high_entry_speed / total_zones
    loss_ratio   = high_speed_loss  / total_zones

    # Classify based on ratios
    if brake_ratio > 0.7 and loss_ratio > 0.5:
        style      = "aggressive-drift"
        confidence = int((brake_ratio + loss_ratio) / 2 * 100)

    elif brake_ratio > 0.6 and entry_ratio > 0.5:
        style      = "late-braker"
        confidence = int((brake_ratio + entry_ratio) / 2 * 100)

    elif brake_ratio < 0.4 and entry_ratio < 0.4:
        style      = "smooth-entry"
        confidence = int((1 - brake_ratio) * 100)

    elif loss_ratio > 0.4 and brake_ratio > 0.4:
        style      = "drift-balanced"
        confidence = int((loss_ratio + brake_ratio) / 2 * 100)

    else:
        style      = "balanced"
        confidence = 50

    return {
        "style":       style,
        "confidence":  confidence,
        "total_zones": total_zones,
        "avg_time_lost_per_lap": round(
            total_time_lost / max(len(lap_summaries), 1), 2
        )
    }


def print_style(result):
    print(f"\n[Ghost AI] Driver Style")
    print(f"  Style:      {result['style']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  Zones analyzed: {result['total_zones']}")
    print(f"  Avg time lost per lap: {result['avg_time_lost_per_lap']}s")
