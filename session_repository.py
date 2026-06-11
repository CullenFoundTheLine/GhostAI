# session_repository.py
# SessionRepository — knows where files live so the rest of the code doesn't have to
#
# Responsibilities:
#   - Find session CSV files in the data directory
#   - Check whether a saved model exists
#   - Save / load the driver fingerprint (JSON summary of training history)
#   - Print a status summary (used by run.py --status)
#
# Nothing in here trains or predicts.
# It is purely a file manager for the data/ folder.

import os
import glob
import json
from typing import List, Optional, Dict, Any


class SessionRepository:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._model_path       = os.path.join(data_dir, "ghost_model.pkl")
        self._fingerprint_path = os.path.join(data_dir, "driver_fingerprint.json")

    # ------------------------------------------------------------------
    # FILE DISCOVERY
    # ------------------------------------------------------------------

    def find_session_files(self) -> List[str]:
        """
        Return all session_*.csv files sorted oldest-first by modification time.
        Oldest first so the training set is chronologically ordered.
        """
        pattern = os.path.join(self.data_dir, "session_*.csv")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        return files

    def latest_session(self) -> Optional[str]:
        """Return the most recently modified session CSV, or None if none exist."""
        files = self.find_session_files()
        return files[-1] if files else None

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------

    def model_exists(self) -> bool:
        """True if a trained model has been saved to disk."""
        return os.path.exists(self._model_path)

    def model_path(self) -> str:
        return self._model_path

    def delete_model(self):
        """Delete the saved model (used by run.py --retrain)."""
        if os.path.exists(self._model_path):
            os.remove(self._model_path)
            print(f"[Ghost AI] Deleted saved model: {self._model_path}")

    # ------------------------------------------------------------------
    # DRIVER FINGERPRINT
    # ------------------------------------------------------------------

    def save_fingerprint(self, fingerprint: Dict[str, Any]):
        """
        Save the driver fingerprint JSON.
        Called by pipeline.py after training to record the driver's style profile.

        fingerprint keys:
          driver          — driver name string
          total_laps      — how many laps were in the training set
          style_breakdown — dict of {style_name: lap_count}
          dominant_style  — the most-seen style name
        """
        os.makedirs(self.data_dir, exist_ok=True)
        with open(self._fingerprint_path, 'w') as f:
            json.dump(fingerprint, f, indent=2)
        print(f"[Ghost AI] Driver fingerprint saved to: {self._fingerprint_path}")

    def load_fingerprint(self) -> Optional[Dict[str, Any]]:
        """
        Load the driver fingerprint from disk.
        Returns None if no fingerprint has been saved yet.
        """
        if not os.path.exists(self._fingerprint_path):
            return None
        with open(self._fingerprint_path, 'r') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def status(self):
        """
        Print a summary of what's in the data directory.
        Called by run.py --status.
        """
        files = self.find_session_files()
        print(f"\n[Ghost AI] === STATUS ===")
        print(f"  Data directory: {self.data_dir}/")
        print(f"  Session files:  {len(files)}")

        for f in files:
            size = os.path.getsize(f)
            # Count data rows (lines minus header)
            try:
                with open(f) as fh:
                    n_rows = sum(1 for _ in fh) - 1
            except Exception:
                n_rows = 0
            print(f"    {os.path.basename(f):40s}  {n_rows:>6} rows  ({size:,} bytes)")

        model_status = "EXISTS" if self.model_exists() else "not found (run: python run.py)"
        print(f"\n  Saved model:    {model_status}")

        fp = self.load_fingerprint()
        if fp:
            dominant = fp.get('dominant_style', '?').upper()
            total    = fp.get('total_laps', 0)
            print(f"  Fingerprint:    EXISTS — {total} laps trained, dominant style: {dominant}")
        else:
            print(f"  Fingerprint:    not found (train first)")

        print()
