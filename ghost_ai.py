# ghostai/ghost_ai.py
# GhostAI — RandomForest model
#
# How this connects to the other files:
#
#   receiver.py  →  writes CSV rows from your PS5
#   coach.py     →  loads those CSVs, extracts 12 features per lap,
#                   calls ghost_ai.learn(X, y) and ghost_ai.predict(x)
#   ghost_ai.py  →  receives those numbers, trains the forest,
#                   classifies new laps, saves/loads the trained model
#
# ghost_ai.py never touches CSV files or GT7 directly.
# It only sees numpy arrays of numbers from coach.py.

import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Tuple, Any, List, Optional

# These match FEATURE_COLUMNS in coach.py exactly.
# The order is critical — index 0 here must be index 0 in every
# feature vector that coach.py builds.
FEATURE_NAMES = [
    'drift_throttle_events',      # full throttle while speed not rising
    'trail_brake_events',         # throttle + brake simultaneously
    'coast_percentage',           # % of lap neither pedal pressed
    'avg_brake_entry_speed',      # how fast you're going when you first brake
    'avg_apex_speed',             # slowest speed through corner
    'n_braking_zones',            # how many distinct braking events
    'n_unexpected_drops',         # instability / lockup events
    'time_loss_score',            # weighted measure of braking duration
    'mean_throttle',              # average throttle across the lap
    'mean_brake',                 # average brake across the lap
    'max_speed_kph',              # top speed reached
    'mean_speed_kph',             # average speed across the lap
]

# Style label numbers → names
# coach.py encodes styles as integers before passing to learn()
# ghost_ai.py decodes them back here
STYLE_LABELS = {
    0: 'aggressive-drift',
    1: 'late-braker',
    2: 'smooth-entry',
    3: 'drift-balanced',
    4: 'balanced',
}

# Default path to save/load the trained model
DEFAULT_MODEL_PATH = "data/ghost_model.pkl"


class GhostAI:
    """
    The RandomForest brain of Ghost AI.

    Receives feature vectors from coach.py,
    trains on them, predicts styles, and explains its decisions.

    Modes:
      "regression"     — predicts a number (time loss score, lap time)
      "classification" — predicts a driving style category label
    """

    def __init__(self, mode: str = "regression"):
        self.mode = mode
        self.model = None
        self.threshold = 0.93
        self.feature_names: List[str] = FEATURE_NAMES
        self.classes_seen: List[int] = []

        # Training memory — stored so you can inspect it after training
        self.training_sample_count: int = 0
        self.feature_importances_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------

    def learn(self, X: np.ndarray, y: np.ndarray):
        """
        Train the RandomForest on feature vectors from coach.py.

        X — shape (n_laps, 12). Each row is one lap's 12 behavioral metrics.
        y — shape (n_laps,).  Each value is the style label (int) or target (float).

        After training, prints:
          1. How many laps were used
          2. Feature importance — which of the 12 metrics mattered most
          3. Style distribution — how many laps of each style were seen
          4. The single behavior that defines you most

        Feature importance is the key output here.
        If 'drift_throttle_events' has importance 0.42, it means
        42% of every decision the forest makes is based on that one number.
        That's Ghost telling you what your driving identity is in the data.
        """
        if self.mode == "regression":
            self.model = RandomForestRegressor(n_estimators=100, random_state=0)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=0)

        self.model.fit(X, y)
        self.training_sample_count = len(X)
        self.feature_importances_ = self.model.feature_importances_

        # --- Print what the forest learned ---
        print("\n[Ghost AI] === LEARNING COMPLETE ===")
        print(f"[Ghost AI] Laps trained on:  {len(X)}")
        print(f"[Ghost AI] Features per lap: {X.shape[1]}")

        # Rank features by importance
        ranked = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda pair: pair[1],
            reverse=True
        )

        print(f"\n[Ghost AI] What the forest learned from your driving:")
        print(f"[Ghost AI] (higher = this behavior defined you more)\n")
        for name, score in ranked:
            bar = "█" * int(score * 40)
            marker = " ← most defining" if name == ranked[0][0] else ""
            print(f"  {name:<30} {bar} {score:.3f}{marker}")

        top_feature = ranked[0][0]
        top_score = ranked[0][1]
        print(f"\n[Ghost AI] Primary driver signature: '{top_feature}'")
        print(f"[Ghost AI] It drove {top_score:.0%} of every classification decision.")
        print(f"[Ghost AI] This is the behavior Ghost reads as most YOU.")

        # Style distribution (classification only)
        if self.mode == "classification":
            self.classes_seen = list(self.model.classes_)
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n[Ghost AI] Styles seen in training data:")
            for label, count in zip(unique, counts):
                style_name = STYLE_LABELS.get(int(label), str(label))
                bar = "█" * count
                print(f"  {style_name:<20} {bar} ({count} lap{'s' if count != 1 else ''})")

            if len(unique) < 3:
                print(f"\n[Ghost AI] NOTE: Only {len(unique)} style(s) in training data.")
                print(f"[Ghost AI] Drive more varied sessions to improve classification.")

        # Target range (regression only)
        if self.mode == "regression":
            print(f"\n[Ghost AI] Target value range:")
            print(f"  Min:  {y.min():.2f}")
            print(f"  Max:  {y.max():.2f}")
            print(f"  Mean: {y.mean():.2f}")

        print(f"\n[Ghost AI] Model ready.")
        print(f"[Ghost AI] =====================================\n")

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------

    def test(self, X: np.ndarray) -> np.ndarray:
        """Run predictions on a batch of feature vectors."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call learn() first.")
        return self.model.predict(X)

    def predict(self, x: np.ndarray) -> Any:
        """Predict a single lap's features. Returns style label or value."""
        return self.test(np.atleast_2d(x))[0]

    def predict_with_confidence(self, x: np.ndarray) -> Tuple[Any, float, str]:
        """
        Predict a single lap AND return how confident the model is.

        For classification: confidence = % of the 100 trees that agreed.
          90%+ = very confident. This lap clearly fits one style.
          60-89% = moderately confident. Mixed signals in the data.
          Below 60% = low confidence. Not enough similar laps in training.

        For regression: confidence is approximated from tree variance.
          Low variance = trees agreed = more reliable prediction.

        Returns: (prediction, confidence_0_to_1, confidence_label)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call learn() first.")

        x2d = np.atleast_2d(x)

        if self.mode == "classification":
            # predict_proba gives probability per class across all trees
            proba = self.model.predict_proba(x2d)[0]
            confidence = float(np.max(proba))
            prediction = self.model.classes_[int(np.argmax(proba))]

        else:
            # For regression: get predictions from each tree individually
            tree_preds = np.array([
                tree.predict(x2d)[0]
                for tree in self.model.estimators_
            ])
            prediction = float(np.mean(tree_preds))
            std = float(np.std(tree_preds))
            # Normalize confidence: low std relative to mean = high confidence
            if prediction != 0:
                confidence = max(0.0, 1.0 - (std / (abs(prediction) + 1e-9)))
            else:
                confidence = 0.5

        # Label the confidence level
        if confidence >= 0.90:
            label = "very confident"
        elif confidence >= 0.70:
            label = "moderately confident"
        elif confidence >= 0.50:
            label = "low confidence — drive more laps"
        else:
            label = "uncertain — not enough similar laps in training"

        return prediction, confidence, label

    def explain_prediction(self, x: np.ndarray) -> str:
        """
        For one lap's feature vector, explain WHY the model made its prediction.

        Shows the top 3 features that drove the decision for THIS specific lap.
        Value × importance = how much that feature influenced the output.

        This is the 'why' answer: not just what style, but which numbers
        in your driving caused that classification.
        """
        if self.model is None:
            return "Model not trained yet."

        x_flat = np.atleast_2d(x)[0]
        prediction, confidence, conf_label = self.predict_with_confidence(x_flat)

        if self.mode == "classification":
            style = STYLE_LABELS.get(int(prediction), str(prediction))
            lines = [
                f"Predicted style:  {style.upper()}",
                f"Confidence:       {confidence:.0%} ({conf_label})",
                f"",
                f"Why Ghost classified this lap this way:",
            ]
        else:
            lines = [
                f"Predicted value:  {prediction:.2f}",
                f"Confidence:       {confidence:.0%} ({conf_label})",
                f"",
                f"What drove this prediction:",
            ]

        importances = self.model.feature_importances_

        # Weight each feature by its value × importance
        # High value * high importance = this feature drove the decision
        weighted = sorted(
            [
                (name, float(val), float(importances[i]))
                for i, (name, val) in enumerate(zip(self.feature_names, x_flat))
            ],
            key=lambda triple: triple[1] * triple[2],
            reverse=True
        )

        for name, val, imp in weighted[:3]:
            influence = val * imp
            lines.append(f"  {name:<30} value={val:.1f}  weight={imp:.3f}  influence={influence:.2f}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, bool]:
        """
        Score the model and check if it meets the 93% threshold.
        Prints per-style accuracy for classification mode.
        """
        preds = self.test(X)

        if self.mode == "classification":
            score = float((preds == y).mean())
            print("[Ghost AI] Per-style accuracy:")
            for label in np.unique(y):
                mask = y == label
                correct = int(np.sum(preds[mask] == y[mask]))
                total = int(np.sum(mask))
                style_name = STYLE_LABELS.get(int(label), str(label))
                pct = correct / total if total > 0 else 0
                print(f"  {style_name:<20} {correct}/{total}  ({pct:.0%})")

        else:
            mae = float(np.mean(np.abs(preds - y)))
            baseline = float(np.mean(np.abs(y - y.mean()))) + 1e-9
            score = max(0.0, 1.0 - (mae / baseline))
            print(f"[Ghost AI] Mean absolute error: {mae:.3f}")
            print(f"[Ghost AI] Regression score:    {score:.2%}")

        meets = score >= self.threshold
        status = "PASSED" if meets else f"needs more data (target: {self.threshold:.0%})"
        print(f"[Ghost AI] Overall score: {score:.2%} — {status}\n")
        return score, meets

    # ------------------------------------------------------------------
    # SAVE / LOAD  (the missing piece — model persists between sessions)
    # ------------------------------------------------------------------

    def save(self, path: str = DEFAULT_MODEL_PATH):
        """
        Save the trained model to disk.

        Without this, every time you run coach.py it retrains from scratch
        and forgets everything. With this, Ghost accumulates knowledge
        across sessions — the more you drive, the smarter it gets
        without having to retrain every time.

        Saves: the model, feature names, classes seen, training count.
        """
        if self.model is None:
            print("[Ghost AI] Nothing to save — model not trained yet.")
            return

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        payload = {
            'mode': self.mode,
            'model': self.model,
            'feature_names': self.feature_names,
            'classes_seen': self.classes_seen,
            'training_sample_count': self.training_sample_count,
            'feature_importances': self.feature_importances_,
            'threshold': self.threshold,
        }

        with open(path, 'wb') as f:
            pickle.dump(payload, f)

        print(f"[Ghost AI] Model saved to: {path}")
        print(f"[Ghost AI] Trained on {self.training_sample_count} laps — "
              f"load this next session instead of retraining.")

    def load(self, path: str = DEFAULT_MODEL_PATH) -> bool:
        """
        Load a previously saved model from disk.

        Call this at the start of coach.py before trying to predict.
        If the file exists, Ghost remembers everything from past sessions.
        If it doesn't exist, you need to train first.

        Returns True if loaded successfully, False if file not found.
        """
        if not os.path.exists(path):
            print(f"[Ghost AI] No saved model at {path}.")
            print(f"[Ghost AI] Call learn() to train a new model.")
            return False

        with open(path, 'rb') as f:
            payload = pickle.load(f)

        self.mode = payload['mode']
        self.model = payload['model']
        self.feature_names = payload['feature_names']
        self.classes_seen = payload['classes_seen']
        self.training_sample_count = payload['training_sample_count']
        self.feature_importances_ = payload['feature_importances']
        self.threshold = payload['threshold']

        print(f"[Ghost AI] Model loaded from: {path}")
        print(f"[Ghost AI] Previously trained on {self.training_sample_count} laps.")
        print(f"[Ghost AI] Ready to predict without retraining.")
        return True

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def status(self):
        """Print current state of the model."""
        print(f"\n[Ghost AI] Status:")
        print(f"  Mode:          {self.mode}")
        print(f"  Trained:       {'yes' if self.model is not None else 'no'}")
        print(f"  Laps seen:     {self.training_sample_count}")
        print(f"  Styles known:  {[STYLE_LABELS.get(c, str(c)) for c in self.classes_seen]}")

        if self.feature_importances_ is not None:
            top = sorted(
                zip(self.feature_names, self.feature_importances_),
                key=lambda p: p[1], reverse=True
            )
            print(f"  Top feature:   {top[0][0]} ({top[0][1]:.3f})")
        print()
