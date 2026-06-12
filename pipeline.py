# pipeline.py
# GhostPipeline — orchestrates the full Ghost AI flow
#
# receive telemetry -> segment laps -> predict style -> generate feedback -> save session
#
# Call order (from run.py):
#   pipeline = GhostPipeline(driver_name="Cullen")
#   pipeline.run()              ← does all 3 steps automatically
#
# Or step by step:
#   pipeline.load_and_analyze() # STEP 1: read CSVs, extract features
#   pipeline.train()            # STEP 2: train RandomForest
#   pipeline.predict_all()      # STEP 3: predict style per lap
#
# Goal evaluation (optional, wired in after GoalEvaluator is ready):
#   pipeline.run_goal_evaluation(csv_path)
 
import os
import numpy as np
from typing import List, Dict, Any, Optional
 
from coach import analyze_file, extract_features, STYLE_MAP
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
        Load all session CSVs, run coach.py on each,
        extract 12 behavioral features per lap.
        Returns total laps loaded.
        """
        print(f"\n[Pipeline] STEP 1 — Loading sessions from {self.data_dir}/...")
        files = self.repo.find_session_files()
 
        if not files:
            print(f"[Pipeline] No session files found in {self.data_dir}/")
            print(f"[Pipeline] Run telemetry/receiver.py first to capture data.")
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
        Train GhostAI on all loaded lap features.
        Falls back to regression if only one style exists in data.
        Returns True if training succeeded.
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
            print(f"[Pipeline] Only 1 style in training data — using regression mode.")
            self.ghost = GhostAI(mode="regression")
            targets = np.array([
                r["summary"].get("time_loss_score", 0.0) for r in self.lap_records
            ])
            self.ghost.learn(X, targets)
        else:
            self.ghost.learn(X, y)
 
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
        Predict the style of every loaded lap.
        Returns list of prediction dicts with confidence and explanation.
        """
        print(f"\n[Pipeline] STEP 3 — Predicting all laps...")
        results = []
 
        for lap in self.lap_records:
            result = self._predict_one(lap["features"], lap["summary"])
            result["file"] = lap["file"]
            result["lap"]  = lap["lap"]
            result["actual_style"] = lap["style"]
            results.append(result)
 
            match = "✓" if result["predicted_style"] == lap["style"] else "✗"
            print(f"  {lap['file']} Lap {lap['lap']} | "
                  f"Predicted: {result['predicted_style']:<20} "
                  f"Confidence: {result['confidence']:.0%} {match}")
 
        return results
 
    def predict_new(self, csv_path: str) -> Dict[str, Any]:
        """
        Predict style for a new session CSV the model hasn't seen.
        Main inference path after receiver.py captures a session.
        """
        print(f"\n[Pipeline] Predicting new session: {csv_path}")
 
        if not self.is_trained and not self._try_load_model():
            print("[Pipeline] No trained model. Run the full pipeline first.")
            return {}
 
        report = analyze_file(csv_path)
        output = {"file": csv_path, "laps": {}}
 
        for lap_no, lap_summary in report["laps"].items():
            features = extract_features(lap_summary)
            if features is None:
                continue
            result = self._predict_one(features, lap_summary)
            output["laps"][lap_no] = result
 
            print(f"\n  Lap {lap_no}:")
            print(f"    Style:      {result['predicted_style'].upper()}")
            print(f"    Confidence: {result['confidence']:.0%} ({result['confidence_label']})")
            print(f"    Why:")
            for line in result["explanation_lines"]:
                print(f"      {line}")
            print(f"    Feedback:")
            for line in result["feedback"]:
                print(f"      {line}")
 
        return output
 
    # ------------------------------------------------------------------
    # FULL PIPELINE RUN
    # ------------------------------------------------------------------
 
    def run(self):
        """
        Run the complete pipeline: load → train → predict → report.
        Loads existing model if saved so it doesn't retrain every time.
        """
        print(f"\n[Ghost AI] Driver: {self.driver_name}")
        print(f"[Ghost AI] Data directory: {self.data_dir}")
        print(f"[Ghost AI] =========================================")
 
        model_loaded = self._try_load_model()
 
        n_laps = self.load_and_analyze()
        if n_laps == 0:
            print("\n[Ghost AI] No data to work with.")
            print("[Ghost AI] Run telemetry/receiver.py to capture a session, then try again.")
            return
 
        if not model_loaded:
            trained = self.train()
            if not trained:
                print("\n[Ghost AI] Could not train model. Drive more sessions.")
                return
        else:
            print(f"\n[Pipeline] Using existing model "
                  f"(trained on {self.ghost.training_sample_count} laps).")
            self.is_trained = True
 
        results = self.predict_all()
        self._print_report(results)
 
    # ------------------------------------------------------------------
    # GOAL EVALUATION — wired in once GoalEvaluator is ready
    # ------------------------------------------------------------------
 
    def run_goal_evaluation(self, csv_path: str) -> list:
        """
        Evaluate braking events in a session CSV against sub-goals.
        Returns the learning set (events with brake_pressure > 200).
 
        Requires: analyzer/goal_evaluator.py to exist.
        Call this after run() or predict_new() — it's a separate pass.
        """
        try:
            from analyzer.goal_evaluator import GoalEvaluator
        except ImportError:
            print("[Pipeline] GoalEvaluator not found. Build analyzer/goal_evaluator.py first.")
            return []
 
        print(f"\n[Pipeline] Running goal evaluation on: {csv_path}")
        evaluator = GoalEvaluator(driver_id=self.driver_name.lower())
        evaluator.evaluate_session(csv_path)
        evaluator.print_summary()
        learning_set = evaluator.get_learning_set()
        print(f"[Pipeline] Learning set: {len(learning_set)} events ready for model")
        return learning_set
 
    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
 
    def _predict_one(self, features: np.ndarray, summary: Dict) -> Dict[str, Any]:
        """Run inference on one lap. Returns style, confidence, explanation, feedback."""
        prediction, confidence, conf_label = self.ghost.predict_with_confidence(features)
 
        if self.ghost.mode == "classification":
            predicted_style = STYLE_LABELS.get(int(prediction), str(prediction))
        else:
            nearest_int = int(round(float(prediction)))
            nearest_int = max(0, min(nearest_int, len(STYLE_LABELS) - 1))
            predicted_style = STYLE_LABELS.get(nearest_int, "balanced")
 
        explanation_text = self.ghost.explain_prediction(features)
        explanation_lines = [l for l in explanation_text.split("\n") if l.strip()]
        feedback = summary.get("feedback", ["No feedback available."])
 
        return {
            "predicted_style":   predicted_style,
            "confidence":        confidence,
            "confidence_label":  conf_label,
            "explanation_lines": explanation_lines,
            "feedback":          feedback,
        }
 
    def _try_load_model(self) -> bool:
        """Load saved model from disk. Returns True if successful."""
        success = self.ghost.load(self._model_path)
        if success:
            self.is_trained = True
        return success
 
    def _print_report(self, results: List[Dict]):
        """Print final session report."""
        if not results:
            return
 
        print(f"\n[Ghost AI] ===== SESSION REPORT — {self.driver_name} =====")
        print(f"[Ghost AI] Total laps analyzed: {len(results)}")
 
        style_counts: Dict[str, int] = {}
        confidences = []
        correct = 0
 
        for r in results:
            s = r["predicted_style"]
            style_counts[s] = style_counts.get(s, 0) + 1
            confidences.append(r["confidence"])
            if r.get("actual_style") == s:
                correct += 1
 
        accuracy = correct / len(results) if results else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
 
        print(f"[Ghost AI] Model accuracy:      {accuracy:.0%}")
        print(f"[Ghost AI] Avg confidence:      {avg_conf:.0%}")
        print(f"\n[Ghost AI] Style breakdown:")
        for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * count
            print(f"  {style:<20} {bar} ({count} lap{'s' if count != 1 else ''})")
 
        dominant = max(style_counts, key=style_counts.get) if style_counts else "unknown"
        print(f"\n[Ghost AI] Driver profile: {self.driver_name.upper()} → {dominant.upper()}")
        print(f"[Ghost AI] ============================================\n")