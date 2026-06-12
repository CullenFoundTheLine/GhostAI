# coach.py
# THE ANALYZER — reads raw DataFrames, extracts behavioral features
# This is what coach.py actually is: an analysis library.
# pipeline.py calls these functions. Nothing here runs on its own.

import numpy as np
import pandas as pd
import os
import glob
import json
from typing import List, Dict, Any, Optional

# Style map — shared with pipeline.py and ghost_ai.py
STYLE_MAP = {
    'aggressive-drift': 0,
    'late-braker':      1,
    'smooth-entry':     2,
    'drift-balanced':   3,
    'balanced':         4,
}
STYLE_MAP_REVERSE = {v: k for k, v in STYLE_MAP.items()}

# Must match FEATURE_NAMES in ghost_ai.py exactly
FEATURE_COLUMNS = [
    'drift_throttle_events',
    'trail_brake_events',
    'coast_percentage',
    'avg_brake_entry_speed',
    'avg_apex_speed',
    'n_braking_zones',
    'n_unexpected_drops',
    'time_loss_score',
    'mean_throttle',
    'mean_brake',
    'max_speed_kph',
    'mean_speed_kph',
]


def load_session(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for c in ['speed_kph', 'throttle', 'brake', 'rpm', 'pos_x', 'pos_y', 'pos_z',
              'tire_fl', 'tire_fr', 'tire_rl', 'tire_rr']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'lap' in df.columns:
        df = df[df['lap'] >= 1]
    return df.reset_index(drop=True)


def segment_laps(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    if 'lap' not in df.columns:
        return {0: df}
    laps = {}
    for lap, group in df.groupby('lap', sort=True):
        laps[int(lap)] = group.reset_index(drop=True)
    return laps


def _find_contiguous_true(mask: np.ndarray) -> List[tuple]:
    runs = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def detect_braking_zones(df, brake_thresh=50.0, speed_drop_thresh=2.0, min_len=3):
    speed = df['speed_kph'].ffill().to_numpy()
    brake = df['brake'].fillna(0).to_numpy()
    throttle = df['throttle'].fillna(0).to_numpy()
    speed_diff = np.concatenate(([0.0], np.diff(speed)))
    mask = (brake > brake_thresh) | ((throttle < 20) & (speed_diff < -speed_drop_thresh))
    runs = _find_contiguous_true(mask)
    zones = []
    for s, e in runs:
        if (e - s + 1) < min_len:
            continue
        zones.append({
            "start_idx": int(s), "end_idx": int(e),
            "start_speed": round(float(speed[s]), 2),
            "end_speed": round(float(speed[e]), 2),
            "speed_drop_kph": round(float(speed[s] - speed[e]), 2),
            "avg_brake": round(float(np.mean(brake[s:e+1])), 2),
            "max_brake": round(float(np.max(brake[s:e+1])), 2),
            "samples": int(e - s + 1),
        })
    return zones


def detect_unexpected_speed_drops(df, drop_thresh=8.0):
    speed = df['speed_kph'].ffill().to_numpy()
    brake = df['brake'].fillna(0).to_numpy()
    speed_diff = np.concatenate(([0.0], np.diff(speed)))
    events = []
    for i in range(1, len(speed_diff)):
        if speed_diff[i] < -drop_thresh and brake[i] < 30:
            events.append({
                "idx": int(i),
                "speed_before": float(speed[i-1]),
                "speed_after": float(speed[i]),
                "delta_kph": round(float(speed_diff[i]), 2),
                "brake": float(brake[i]),
            })
    return events


def detect_drift_events(df, throttle_thresh=200.0):
    throttle = df['throttle'].fillna(0).to_numpy()
    speed = df['speed_kph'].ffill().to_numpy()
    speed_diff = np.concatenate(([0.0], np.diff(speed)))
    return int(np.sum((throttle > throttle_thresh) & (speed_diff <= 0)))


def detect_trail_braking(df, min_throttle=20.0, min_brake=20.0):
    throttle = df['throttle'].fillna(0).to_numpy()
    brake = df['brake'].fillna(0).to_numpy()
    return int(np.sum((throttle > min_throttle) & (brake > min_brake)))


def coast_percentage(df, max_throttle=10.0, max_brake=10.0):
    throttle = df['throttle'].fillna(0).to_numpy()
    brake = df['brake'].fillna(0).to_numpy()
    coasting = np.sum((throttle < max_throttle) & (brake < max_brake))
    return round(float(coasting / max(len(df), 1)) * 100, 1)


def summarize_lap(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {'n_samples': int(len(df))}
    if len(df) == 0:
        return summary

    summary['mean_speed_kph'] = float(df['speed_kph'].mean())
    summary['max_speed_kph']  = float(df['speed_kph'].max())
    summary['min_speed_kph']  = float(df['speed_kph'].min())
    summary['mean_throttle']  = float(df['throttle'].mean())
    summary['mean_brake']     = float(df['brake'].mean())

    zones  = detect_braking_zones(df)
    drops  = detect_unexpected_speed_drops(df)
    drift  = detect_drift_events(df)
    trail  = detect_trail_braking(df)
    coast  = coast_percentage(df)

    summary['braking_zones']           = zones
    summary['n_braking_zones']         = len(zones)
    summary['unexpected_speed_drops']  = drops
    summary['n_unexpected_drops']      = len(drops)
    summary['drift_throttle_events']   = drift
    summary['trail_brake_events']      = trail
    summary['coast_percentage']        = coast

    summary['avg_brake_entry_speed'] = (
        round(float(np.mean([z['start_speed'] for z in zones])), 2)
        if zones else 0.0
    )
    summary['avg_apex_speed'] = round(float(df['speed_kph'].min()), 2)

    time_loss = sum(z['speed_drop_kph'] * (z['samples'] / 10.0) for z in zones)
    summary['time_loss_score'] = round(time_loss, 2)

    style   = classify_style(summary)
    feedback = generate_feedback(summary, style)
    summary['style']    = style
    summary['feedback'] = feedback
    return summary


def classify_style(summary: Dict[str, Any]) -> str:
    drift  = summary.get('drift_throttle_events', 0)
    trail  = summary.get('trail_brake_events', 0)
    coast  = summary.get('coast_percentage', 0)
    apex   = summary.get('avg_apex_speed', 100)
    if drift > 10 and apex < 60:
        return 'aggressive-drift'
    elif trail > 8:
        return 'late-braker'
    elif coast > 35:
        return 'smooth-entry'
    elif drift > 5:
        return 'drift-balanced'
    return 'balanced'


def generate_feedback(summary: Dict[str, Any], style: str) -> List[str]:
    feedback = []
    coast = summary.get('coast_percentage', 0)
    if coast > 30:
        feedback.append(f"GHOST: {coast:.0f}% coast time — apply throttle earlier on exit.")
    drops = summary.get('n_unexpected_drops', 0)
    if drops > 2:
        feedback.append(f"GHOST: {drops} instability events — possible spin or off-track.")
    drift = summary.get('drift_throttle_events', 0)
    if drift > 5:
        feedback.append(f"GHOST: {drift} drift events — rear bias + LSD recommended.")
    trail = summary.get('trail_brake_events', 0)
    if trail > 3:
        feedback.append(f"GHOST: {trail} trail brake events — refine the overlap timing.")
    if not feedback:
        feedback.append("GHOST: Clean lap captured. Keep building sessions.")
    return feedback


def extract_features(lap_summary: Dict[str, Any]) -> Optional[np.ndarray]:
    row = []
    for col in FEATURE_COLUMNS:
        val = lap_summary.get(col, 0)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        row.append(float(val))
    if sum(row) == 0:
        return None
    return np.array(row)


def analyze_file(path: str) -> Dict[str, Any]:
    df = load_session(path)
    laps = segment_laps(df)
    report = {"file": os.path.basename(path), "n_samples": len(df), "laps": {}}
    for lap_no, lap_df in laps.items():
        report["laps"][lap_no] = summarize_lap(lap_df)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default=None)
    args = parser.parse_args()

    if args.file:
        path = args.file
    else:
        files = sorted(glob.glob("data/session_*.csv"), key=os.path.getmtime, reverse=True)
        if not files:
            raise SystemExit("No session files in data/")
        path = files[0]

    rpt = analyze_file(path)
    print(json.dumps(rpt, indent=2))
