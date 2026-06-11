# pipeline.py
# GhostPipeline — full pipeline from raw session CSVs to trained model to predictions
#
# This is what orchestrates everything.
# It does NOT know about GT7, PS5, or files directly —
# it uses SessionRepository to find files and GhostAI to learn/predict.
#
# Typical call order (from run.py):
#   pipeline = GhostPipeline(driver_name="Cullen")
#   pipeline.run()       ← does all 3 steps automatically
#
# Or step by step:
#   pipeline.load_and_analyze()   # STEP 1: read CSVs, extract features
#   pipeline.train()              # STEP 2: train RandomForest
#   pipeline.predict_all()        # STEP 3: predict style per lap

import os
import numpy as np
from typing import List, Dict, Any, Optional

from coach import analyze_file, extract_features, STYLE_MAP, STYLE_MAP_REVERSE
from ghost_ai import GhostAI, STYLE_LABELS
from session_repository import SessionRepository


class GhostPipeline:
    def __init__(self, driver_name: str = "Driver", data_dir: str = "data"):
        self.driver_name = driver_name
        self.data_dir = data_dir
        self.repo = SessionRepository(data_dir)
        self.ghost = GhostAI(mode="classification")
        self.lap_records: List[Dict] = []
        self.is_trained = False
        self._model_path = os.path.join(data_dir, "ghost_model.pkl")

    # ------------------------------------------------------------------
    # STEP 1 — LOAD AND ANALYZE
    # ------------------------------------------------------------------

    def load_and_analyze(self) -> int:
        """
        STEP 1: Load all session CSVs from data_dir, run coach.py on each,
        extract 12 behavioral features per lap, store in self.lap_records.

        Returns total number of laps loaded.
        """
        print(f"\n[Pipeline] STEP 1 — Loading sessions from {self.data_dir}/...")
        files = self.repo.find_session_files()

        if not files:
            print(f"[Pipeline] No session files found in {self.data_dir}/")
            print(f"[Pipeline] Run receiver.py first to capture data.")
            return 0

        self.lap_records = []

        for path in files:
            try:
                report = analyze_file(path)
                laps_in_file = 0
                for lap_no, lap_summary in report["laps"].items():
                    features = extract_features(lap_summary)
                    if features is None:
                        continue
                    style = lap_summary.get("style", "balanced")
                    style_int = STYLE_MAP.get(style, 4)
                    self.lap_records.append({
                        "file":      os.path.basename(path),
                        "lap":       lap_no,
                        "style":     style,
                        "style_int": style_int,
                        "features":  features,
                        "summary":   lap_summary,
                    })
                    laps_in_file += 1
                print(f"  Loaded: {os.path.basename(path)} ({laps_in_file} laps)")
            except Exception as e:
                print(f"  Skipped {os.path.basename(path)}: {e}")

        print(f"[Pipeline] Total laps loaded: {len(self.lap_records)}")
        return len(self.lap_records)

    # ------------------------------------------------------------------
    # STEP 2 — TRAIN
    # ------------------------------------------------------------------

    def train(self) -> bool:
        """
        STEP 2: Train GhostAI on all loaded lap features.

        If only one driving style exists in the data, falls back to regression
        (predicts time_loss_score) because a classifier needs at least 2 classes.

        Returns True if training succeeded, False otherwise.
        """
        print(f"\n[Pipeline] STEP 2 — Training GhostAI...")

        if not self.lap_records:
            print("[Pipeline] No lap data to train on. Run load_and_analyze() first.")
            return False

        if len(self.lap_records) < 2:
            print(f"[Pipeline] Need at least 2 laps to train. Have {len(self.lap_records)}.")
            return False

        X = np.array([r["features"] for r in self.lap_records])
        y = np.array([r["style_int"] for r in self.lap_records])

        unique_styles = len(np.unique(y))
        if unique_styles < 2:
            # Only one style seen — switch to regression so the model is still useful
            print(f"[Pipeline] Only 1 style in training data — using regression mode.")
            self.ghost = GhostAI(mode="regression")
            targets = np.array([
                r["summary"].get("time_loss_score", 0.0) for r in self.lap_records
            ])
            self.ghost.learn(X, targets)
        else:
            self.ghost.learn(X, y)

        # Persist the trained model to disk
        self.ghost.save(self._model_path)
        self.is_trained = True

        # Build and save driver fingerprint
        style_counts: Dict[str, int] = {}
        for r in self.lap_records:
            style_counts[r["style"]] = style_counts.get(r["style"], 0) + 1

        dominant = max(style_counts, key=style_counts.get)
        fingerprint = {
            "driver":          self.driver_name,
            "total_laps":      len(self.lap_records),
            "style_breakdown": style_counts,
            "dominant_style":  dominant,
        }
        self.repo.save_fingerprint(fingerprint)
        return True

    # ------------------------------------------------------------------
    # STEP 3 — PREDICT
    # ------------------------------------------------------------------

    def predict_all(self) -> List[Dict]:
        """
        STEP 3: Predict the style of every loaded lap.
        Returns predictions with confidence and explanation.
        """
        print(f"\n[Pipeline] STEP 3 — Predicting all laps...")
        results = []

        for lap in self.lap_records:
            result = self._predict_one(lap['features'], lap['summary'])
            result['file'] = lap['file']
            result['lap'] = lap['lap']
            result['actual_style'] = lap['style']
            results.append(result)

            match = "✓" if result['predicted_style'] == lap['style'] else "✗"
            print(f"  {lap['file']} Lap {lap['lap']} | "
                  f"Predicted: {result['predicted_style']:<20} "
                  f"Confidence: {result['confidence']:.0%} {match}")

        return results

    def predict_new(self, csv_path: str) -> Dict[str, Any]:
        """
        Predict style for a brand new session CSV the model hasn't seen.
        This is the main inference path — receiver.py captures a session,
        you pass it here, Ghost tells you what style it detected.
        """
        print(f"\n[Pipeline] Predicting new session: {csv_path}")

        if not self.is_trained and not self._try_load_model():
            print("[Pipeline] No trained model. Run the full pipeline first.")
            return {}

        report = analyze_file(csv_path)
        output = {'file': csv_path, 'laps': {}}

        for lap_no, lap_summary in report["laps"].items():
            features = extract_features(lap_summary)
            if features is None:
                continue
            result = self._predict_one(features, lap_summary)
            output['laps'][lap_no] = result

            print(f"\n  Lap {lap_no}:")
            print(f"    Style:      {result['predicted_style'].upper()}")
            print(f"    Confidence: {result['confidence']:.0%} ({result['confidence_label']})")
            print(f"    Why:")
            for line in result['explanation_lines']:
                print(f"      {line}")
            print(f"    Feedback:")
            for line in result['feedback']:
                print(f"      {line}")

        return output

    # ------------------------------------------------------------------
    # FULL PIPELINE (called by run.py default command)
    # ------------------------------------------------------------------

    def run(self):
        """
        Run the complete pipeline: load → train → predict → report.
        Loads an existing model if one is saved so it doesn't retrain every time.
        """
        print(f"\n[Ghost AI] Driver: {self.driver_name}")
        print(f"[Ghost AI] Data directory: {self.data_dir}")
        print(f"[Ghost AI] =========================================")

        # Try to load a previously saved model first
        model_loaded = self._try_load_model()

        # Always re-load and analyze the latest session data
        n_laps = self.load_and_analyze()
        if n_laps == 0:
            print("\n[Ghost AI] No data to work with.")
            print("[Ghost AI] Run receiver.py to capture a session, then try again.")
            return

        # Train if no model was found (or if it was deleted via --retrain)
        if not model_loaded:
            trained = self.train()
            if not trained:
                print("\n[Ghost AI] Could not train model. Drive more sessions.")
                return
        else:
            print(f"\n[Pipeline] Using existing model "
                  f"(trained on {self.ghost.training_sample_count} laps).")
            self.is_trained = True

        # Predict style for every loaded lap
        results = self.predict_all()

        # Print the final report
        self._print_report(results)

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _predict_one(self, features: np.ndarray, summary: Dict) -> Dict[str, Any]:
        """
        Run inference on a single lap's feature vector.
        Returns a dict with predicted style, confidence, explanation, and feedback.
        """
        prediction, confidence, conf_label = self.ghost.predict_with_confidence(features)

        # Map numeric prediction back to a human-readable style name
        if self.ghost.mode == "classification":
            predicted_style = STYLE_LABELS.get(int(prediction), str(prediction))
        else:
            # Regression: find the nearest style by integer label
            nearest_int = int(round(float(prediction)))
            nearest_int = max(0, min(nearest_int, len(STYLE_LABELS) - 1))
            predicted_style = STYLE_LABELS.get(nearest_int, "balanced")

        # Get the "why" explanation from GhostAI
        explanation_text = self.ghost.explain_prediction(features)
        explanation_lines = [
            line for line in explanation_text.split('\n') if line.strip()
        ]

        # Feedback was generated by coach.py during analysis
        feedback = summary.get('feedback', ["No feedback available."])

        return {
            'predicted_style':   predicted_style,
            'confidence':        confidence,
            'confidence_label':  conf_label,
            'explanation_lines': explanation_lines,
            'feedback':          feedback,
        }

    def _try_load_model(self) -> bool:
        """
        Attempt to load a previously saved model from disk.
        Returns True if a model was loaded, False if none exists.
        """
        success = self.ghost.load(self._model_path)
        if success:
            self.is_trained = True
        return success

    def _print_report(self, results: List[Dict]):
        """Print the final session report after predict_all()."""
        if not results:
            return

        print(f"\n[Ghost AI] ===== SESSION REPORT — {self.driver_name} =====")
        print(f"[Ghost AI] Total laps analyzed: {len(results)}")

        style_counts: Dict[str, int] = {}
        confidences = []
        correct = 0

        for r in results:
            s = r['predicted_style']
            style_counts[s] = style_counts.get(s, 0) + 1
            confidences.append(r['confidence'])
            if r.get('actual_style') == s:
                correct += 1

        accuracy  = correct / len(results) if results else 0.0
        avg_conf  = sum(confidences) / len(confidences) if confidences else 0.0

        print(f"[Ghost AI] Model accuracy:      {accuracy:.0%}")
        print(f"[Ghost AI] Avg confidence:      {avg_conf:.0%}")
        print(f"\n[Ghost AI] Style breakdown:")
        for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * count
            print(f"  {style:<20} {bar} ({count} lap{'s' if count != 1 else ''})")

        dominant = max(style_counts, key=style_counts.get) if style_counts else "unknown"
        print(f"\n[Ghost AI] Driver profile: {self.driver_name.upper()} → {dominant.upper()}")
        print(f"[Ghost AI] ============================================\n")
