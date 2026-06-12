# analyzer/goal_evaluator.py
#
# GoalEvaluator — Constructor class for braking zone target and sub-goal evaluation
#
# Joseph's requirement:
#   For every brake point event:
#     if brake_pressure > 200 → save and learn
#     else → discard
#
# Extended with graduated sub-goals so Ghost AI knows
# HOW FAR from the target the driver is, not just pass/fail

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


# ── Sub-goal: one graduated threshold level ────────────────────────
@dataclass
class SubGoal:
    """
    A single performance threshold within a target goal.
    Sub-goals are ordered from easiest to hardest.
    Ghost AI uses these to tell the driver exactly where they stand.
    """
    name: str                    # e.g. "Entry", "Developing", "Optimal"
    min_brake_pressure: float    # minimum brake value (0-255 scale)
    min_duration_s: float        # minimum brake duration in seconds
    description: str             # plain English explanation


# ── Target goal: one braking zone objective ────────────────────────
@dataclass
class TargetGoal:
    """
    A target goal defines what Ghost AI is trying to learn
    and evaluate for a specific braking zone type.

    Examples:
      - "Heavy braking from high speed"
      - "Trail brake into hairpin"
      - "Chicane brake"
    """
    name: str
    sub_goals: List[SubGoal] = field(default_factory=list)
    description: str = ""

    def evaluate(self, brake_pressure: float, duration_s: float) -> Optional[SubGoal]:
        """
        Given a brake event, return the highest sub-goal it meets.
        Returns None if it doesn't meet any sub-goal (event is discarded).
        """
        passed = [
            sg for sg in self.sub_goals
            if brake_pressure >= sg.min_brake_pressure
            and duration_s >= sg.min_duration_s
        ]
        if not passed:
            return None
        # Return the hardest sub-goal met
        return max(passed, key=lambda sg: sg.min_brake_pressure)


# ── BrakeEvent result: what happened at one braking zone ──────────
@dataclass
class BrakeEventResult:
    """
    The result of evaluating one braking event against the GoalEvaluator.
    This is what gets saved to the learning set.
    """
    lap: int
    frame_idx: int
    brake_pressure: float
    duration_s: float
    speed_before: float
    goal_name: str
    sub_goal_met: Optional[str]   # None means discarded
    should_learn: bool
    feedback: str


# ── GoalEvaluator: the constructor class ──────────────────────────
class GoalEvaluator:
    """
    Constructor class for evaluating braking events against
    target goals and sub-goals.

    Usage:
        evaluator = GoalEvaluator(driver_id="cullen")
        results = evaluator.evaluate_session(csv_path)
        learning_set = evaluator.get_learning_set()

    Joseph's rule:
        brake_pressure > 200 → save and learn
        else → discard

    Extended:
        Sub-goals give graduated feedback, not just pass/fail.
        The learning set only contains events that passed.
    """

    # ── Joseph's threshold (the core rule) ────────────────────────
    LEARN_THRESHOLD = 200   # minimum brake pressure to save an event

    def __init__(self, driver_id: str = "unknown"):
        self.driver_id = driver_id

        # All evaluated events this session
        self._all_results: List[BrakeEventResult] = []

        # Only events that passed — the learning set
        self._learning_set: List[BrakeEventResult] = []

        # Stats
        self.total_evaluated = 0
        self.total_saved = 0
        self.total_discarded = 0

        # Define the target goals and sub-goals
        self._goals = self._build_goals()

    def _build_goals(self) -> List[TargetGoal]:
        """
        Define target goals and their sub-goals.
        These thresholds are based on real GT7 telemetry (0-255 brake scale).

        YOU define the numbers here — you know what a bad brake zone
        feels like. Joseph structures the pipeline around your numbers.
        """
        return [

            TargetGoal(
                name="heavy_brake",
                description="Hard braking from speed above 150 kph",
                sub_goals=[
                    SubGoal(
                        name="Entry",
                        min_brake_pressure=150,
                        min_duration_s=0.3,
                        description="Touched the brakes — not committed"
                    ),
                    SubGoal(
                        name="Developing",
                        min_brake_pressure=200,
                        min_duration_s=0.5,
                        description="Real braking zone — learnable"
                    ),
                    SubGoal(
                        name="Optimal",
                        min_brake_pressure=230,
                        min_duration_s=0.8,
                        description="Late, hard brake — aggressive fingerprint"
                    ),
                ]
            ),

            TargetGoal(
                name="trail_brake",
                description="Throttle-brake overlap into a corner",
                sub_goals=[
                    SubGoal(
                        name="Developing",
                        min_brake_pressure=50,
                        min_duration_s=0.2,
                        description="Some overlap detected"
                    ),
                    SubGoal(
                        name="Structured",
                        min_brake_pressure=100,
                        min_duration_s=0.4,
                        description="Consistent trail braking technique"
                    ),
                ]
            ),

            TargetGoal(
                name="chicane_brake",
                description="Quick direction change braking",
                sub_goals=[
                    SubGoal(
                        name="Entry",
                        min_brake_pressure=120,
                        min_duration_s=0.2,
                        description="Quick touch"
                    ),
                    SubGoal(
                        name="Optimal",
                        min_brake_pressure=200,
                        min_duration_s=0.3,
                        description="Sharp committed chicane brake"
                    ),
                ]
            ),

        ]

    def _classify_goal(self, brake_pressure: float,
                       duration_s: float,
                       speed_before: float) -> TargetGoal:
        """
        Pick which target goal applies to this brake event
        based on context (speed, duration, pressure).
        """
        if speed_before > 150 and duration_s > 0.5:
            return self._goals[0]  # heavy_brake
        elif duration_s < 0.4 and brake_pressure > 80:
            return self._goals[2]  # chicane_brake
        else:
            return self._goals[1]  # trail_brake

    def evaluate_event(self,
                       brake_pressure: float,
                       duration_s: float,
                       speed_before: float,
                       lap: int = 0,
                       frame_idx: int = 0) -> BrakeEventResult:
        """
        Evaluate a single braking event.

        Joseph's rule:
            if brake_pressure > LEARN_THRESHOLD → save and learn
            else → discard

        Returns a BrakeEventResult with full context.
        """
        self.total_evaluated += 1

        # Pick the appropriate goal
        goal = self._classify_goal(brake_pressure, duration_s, speed_before)

        # Check which sub-goal it meets
        sub_goal = goal.evaluate(brake_pressure, duration_s)

        # Apply Joseph's core rule
        should_learn = brake_pressure > self.LEARN_THRESHOLD

        # Build feedback message
        if not should_learn:
            feedback = (
                f"Brake pressure {brake_pressure:.0f} — below threshold "
                f"({self.LEARN_THRESHOLD}). Event discarded."
            )
            self.total_discarded += 1
        elif sub_goal is None:
            feedback = (
                f"Brake pressure {brake_pressure:.0f} passes threshold "
                f"but didn't meet any sub-goal for '{goal.name}'. Saved anyway."
            )
            self.total_saved += 1
        else:
            feedback = (
                f"Sub-goal MET: [{goal.name}] → {sub_goal.name} "
                f"({sub_goal.description}). Saved."
            )
            self.total_saved += 1

        result = BrakeEventResult(
            lap=lap,
            frame_idx=frame_idx,
            brake_pressure=brake_pressure,
            duration_s=duration_s,
            speed_before=speed_before,
            goal_name=goal.name,
            sub_goal_met=sub_goal.name if sub_goal else None,
            should_learn=should_learn,
            feedback=feedback
        )

        self._all_results.append(result)
        if should_learn:
            self._learning_set.append(result)

        return result

    def evaluate_session(self, csv_path: str) -> List[BrakeEventResult]:
        """
        Read a session CSV and evaluate all braking events.
        This is what pipeline.py calls.
        """
        import pandas as pd
        import os

        if not os.path.exists(csv_path):
            print(f"[GoalEvaluator] File not found: {csv_path}")
            return []

        df = pd.read_csv(csv_path)
        for col in ["brake", "speed_kph", "lap"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "lap" in df.columns:
            df = df[df["lap"] >= 1]

        if df.empty:
            print(f"[GoalEvaluator] No driving data in {csv_path}")
            return []

        # Detect braking events: brake crosses threshold
        results = []
        in_brake = False
        brake_start_idx = 0
        brake_start_pressure = 0
        speed_at_entry = 0
        lap_at_start = 0

        for idx, row in df.iterrows():
            brake = row.get("brake", 0) or 0
            speed = row.get("speed_kph", 0) or 0
            lap = row.get("lap", 0) or 0

            if brake > 30 and not in_brake:
                in_brake = True
                brake_start_idx = idx
                brake_start_pressure = brake
                speed_at_entry = speed
                lap_at_start = lap

            elif brake <= 30 and in_brake:
                in_brake = False
                duration_frames = idx - brake_start_idx
                duration_s = duration_frames / 60.0  # ~60fps

                result = self.evaluate_event(
                    brake_pressure=brake_start_pressure,
                    duration_s=duration_s,
                    speed_before=speed_at_entry,
                    lap=int(lap_at_start),
                    frame_idx=int(brake_start_idx)
                )
                results.append(result)

        return results

    def get_learning_set(self) -> List[BrakeEventResult]:
        """Return only the events that passed — what the model learns from."""
        return self._learning_set

    def summary(self) -> Dict[str, Any]:
        """Print a session summary."""
        sub_goal_counts: Dict[str, int] = {}
        for r in self._learning_set:
            key = r.sub_goal_met or "no_sub_goal"
            sub_goal_counts[key] = sub_goal_counts.get(key, 0) + 1

        return {
            "driver_id": self.driver_id,
            "total_evaluated": self.total_evaluated,
            "total_saved": self.total_saved,
            "total_discarded": self.total_discarded,
            "save_rate_pct": round(
                self.total_saved / max(self.total_evaluated, 1) * 100, 1
            ),
            "sub_goal_breakdown": sub_goal_counts,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n[GoalEvaluator] ── Session Summary ──")
        print(f"  Driver:     {s['driver_id']}")
        print(f"  Evaluated:  {s['total_evaluated']} events")
        print(f"  Saved:      {s['total_saved']} (learning set)")
        print(f"  Discarded:  {s['total_discarded']}")
        print(f"  Save rate:  {s['save_rate_pct']}%")
        print(f"  Sub-goals:")
        for sg, count in s['sub_goal_breakdown'].items():
            print(f"    {sg}: {count}")
        print()