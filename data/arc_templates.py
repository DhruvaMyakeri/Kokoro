"""
Emotional arc templates for synthetic trajectory construction.

Each template defines a sequence of emotional zones in Russell's circumplex space
(valence × arousal, both continuous in [-1, 1]).

Arc templates are grounded in psychological research on emotional trajectories:
- Gradual decline: Seligman (2011) on learned helplessness onset
- Recovery arc: Bonnano (2004) resilience trajectory literature
- Acute stress: Lazarus (1993) stress-appraisal-coping model
- Grief arc: Kübler-Ross / Stroebe dual-process model
- Chronic anxiety: Clark & Beck (2010) generalized anxiety patterns
- Oscillating: Davidson (2000) emotional recovery with weekend variation
- Growth arc: post-traumatic growth literature (Tedeschi & Calhoun, 1996)

No emotion labels are stored in these templates — only continuous circumplex
coordinates and their tolerances. The emotion label→coordinate mapping in
prepare.py is the only place labels appear, and it is used solely to select
training conversations, not to constrain model output.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CircumplexZone:
    """
    A region in Russell's circumplex space.
    valence: [-1, 1] — negative to positive affect
    arousal: [-1, 1] — low to high activation
    tolerance: how wide the sampling window is around the center point
    """
    valence: float
    arousal: float
    tolerance: float = 0.25
    label: str = ""  # human-readable name for debugging only, not used in model

    def contains(self, v: float, a: float) -> bool:
        return (
            abs(v - self.valence) <= self.tolerance
            and abs(a - self.arousal) <= self.tolerance
        )


@dataclass
class ArcTemplate:
    """
    A named sequence of circumplex zones representing a realistic emotional
    trajectory across N sessions.
    """
    name: str
    description: str
    sessions: list[CircumplexZone]
    typical_duration_weeks: tuple[int, int]  # (min, max) weeks this arc spans
    weight: float = 1.0  # sampling weight — more common arcs are sampled more


# ---------------------------------------------------------------------------
# Zone anchors — circumplex positions grounded in Russell (1980) and
# Posner, Russell & Peterson (2005). These are approximate centroids of
# emotion clusters, not discrete labels.
# ---------------------------------------------------------------------------

_ZONES = {
    # Quadrant I: high valence, high arousal
    "elated":      CircumplexZone(valence=+0.8, arousal=+0.7, label="elated"),
    "excited":     CircumplexZone(valence=+0.6, arousal=+0.6, label="excited"),
    "hopeful":     CircumplexZone(valence=+0.5, arousal=+0.3, label="hopeful"),

    # Quadrant II: high valence, low arousal
    "content":     CircumplexZone(valence=+0.5, arousal=-0.3, label="content"),
    "calm":        CircumplexZone(valence=+0.3, arousal=-0.5, label="calm"),
    "grateful":    CircumplexZone(valence=+0.6, arousal=-0.1, label="grateful"),

    # Transition/neutral
    "neutral":     CircumplexZone(valence=0.0,  arousal=0.0,  label="neutral", tolerance=0.3),
    "unsettled":   CircumplexZone(valence=-0.1, arousal=+0.2, label="unsettled"),

    # Quadrant III: low valence, high arousal
    "anxious":     CircumplexZone(valence=-0.4, arousal=+0.5, label="anxious"),
    "distressed":  CircumplexZone(valence=-0.6, arousal=+0.6, label="distressed"),
    "panicked":    CircumplexZone(valence=-0.7, arousal=+0.8, label="panicked"),
    "angry":       CircumplexZone(valence=-0.5, arousal=+0.7, label="angry"),

    # Quadrant IV: low valence, low arousal
    "sad":         CircumplexZone(valence=-0.5, arousal=-0.3, label="sad"),
    "depressed":   CircumplexZone(valence=-0.7, arousal=-0.5, label="depressed"),
    "exhausted":   CircumplexZone(valence=-0.4, arousal=-0.6, label="exhausted"),
    "lonely":      CircumplexZone(valence=-0.6, arousal=-0.2, label="lonely"),
}


# ---------------------------------------------------------------------------
# Arc templates
# ---------------------------------------------------------------------------

ARC_TEMPLATES: list[ArcTemplate] = [

    ArcTemplate(
        name="gradual_decline",
        description=(
            "Slow erosion of positive affect over weeks. Common in burnout, "
            "chronic stress, and unaddressed relationship strain."
        ),
        sessions=[
            _ZONES["content"],
            _ZONES["content"],
            _ZONES["unsettled"],
            _ZONES["unsettled"],
            _ZONES["anxious"],
            _ZONES["anxious"],
            _ZONES["sad"],
            _ZONES["depressed"],
        ],
        typical_duration_weeks=(4, 8),
        weight=1.5,
    ),

    ArcTemplate(
        name="slow_recovery",
        description=(
            "Gradual improvement from a depressed or anxious baseline. "
            "Typical in therapy, social support, or life change. "
            "Non-linear — includes minor setbacks."
        ),
        sessions=[
            _ZONES["depressed"],
            _ZONES["sad"],
            _ZONES["sad"],
            _ZONES["unsettled"],
            _ZONES["neutral"],
            _ZONES["anxious"],    # minor setback
            _ZONES["neutral"],
            _ZONES["hopeful"],
            _ZONES["content"],
        ],
        typical_duration_weeks=(6, 12),
        weight=1.5,
    ),

    ArcTemplate(
        name="acute_stress_stabilization",
        description=(
            "Sudden spike into distress (job loss, relationship crisis, medical news), "
            "followed by gradual stabilization. Arousal drops faster than valence recovers."
        ),
        sessions=[
            _ZONES["content"],
            _ZONES["panicked"],
            _ZONES["distressed"],
            _ZONES["anxious"],
            _ZONES["unsettled"],
            _ZONES["unsettled"],
            _ZONES["neutral"],
        ],
        typical_duration_weeks=(3, 6),
        weight=1.2,
    ),

    ArcTemplate(
        name="chronic_low_grade_anxiety",
        description=(
            "Persistent mild-to-moderate anxiety without acute crisis. "
            "Slight oscillation but no clear trend. GAD-like pattern."
        ),
        sessions=[
            _ZONES["anxious"],
            _ZONES["unsettled"],
            _ZONES["anxious"],
            _ZONES["anxious"],
            _ZONES["unsettled"],
            _ZONES["anxious"],
            _ZONES["distressed"],
            _ZONES["anxious"],
        ],
        typical_duration_weeks=(6, 16),
        weight=1.3,
    ),

    ArcTemplate(
        name="weekend_oscillation",
        description=(
            "Regular weekday stress / weekend relief cycle. Models people with "
            "demanding jobs who decompress on weekends. Mean stays mildly negative."
        ),
        sessions=[
            _ZONES["anxious"],    # weekday
            _ZONES["calm"],       # weekend relief
            _ZONES["anxious"],
            _ZONES["content"],
            _ZONES["distressed"],
            _ZONES["calm"],
            _ZONES["anxious"],
            _ZONES["hopeful"],
        ],
        typical_duration_weeks=(4, 8),
        weight=1.0,
    ),

    ArcTemplate(
        name="grief_arc",
        description=(
            "Acute loss followed by slow, incomplete recovery. "
            "Valence drops sharply, arousal is initially high (shock/anger) "
            "then drops (despair). Recovery is slow and never reaches baseline."
        ),
        sessions=[
            _ZONES["content"],
            _ZONES["panicked"],   # acute shock
            _ZONES["angry"],      # anger stage
            _ZONES["depressed"],  # despair
            _ZONES["depressed"],
            _ZONES["sad"],
            _ZONES["sad"],
            _ZONES["lonely"],
            _ZONES["neutral"],    # partial recovery only
        ],
        typical_duration_weeks=(8, 20),
        weight=1.0,
    ),

    ArcTemplate(
        name="post_traumatic_growth",
        description=(
            "Acute trauma followed by deeper recovery that eventually exceeds "
            "prior baseline — models resilience and meaning-making. "
            "Based on Tedeschi & Calhoun (1996)."
        ),
        sessions=[
            _ZONES["neutral"],
            _ZONES["distressed"],
            _ZONES["depressed"],
            _ZONES["sad"],
            _ZONES["neutral"],
            _ZONES["hopeful"],
            _ZONES["hopeful"],
            _ZONES["grateful"],
            _ZONES["elated"],
        ],
        typical_duration_weeks=(8, 16),
        weight=0.8,
    ),

    ArcTemplate(
        name="social_confidence_growth",
        description=(
            "Gradual increase in positive affect as social support, "
            "achievement, or therapeutic progress accumulates. No crisis event."
        ),
        sessions=[
            _ZONES["unsettled"],
            _ZONES["neutral"],
            _ZONES["neutral"],
            _ZONES["hopeful"],
            _ZONES["hopeful"],
            _ZONES["content"],
            _ZONES["excited"],
        ],
        typical_duration_weeks=(4, 10),
        weight=1.0,
    ),

    ArcTemplate(
        name="relapse_dip",
        description=(
            "Recovery in progress, then a setback that partially undoes gains, "
            "followed by re-stabilization. Common in addiction recovery, depression."
        ),
        sessions=[
            _ZONES["sad"],
            _ZONES["neutral"],
            _ZONES["hopeful"],
            _ZONES["content"],
            _ZONES["distressed"],  # relapse
            _ZONES["sad"],
            _ZONES["unsettled"],
            _ZONES["neutral"],
            _ZONES["hopeful"],
        ],
        typical_duration_weeks=(6, 14),
        weight=1.1,
    ),

    ArcTemplate(
        name="stable_positive",
        description=(
            "Sustained positive affect, minor fluctuations. Healthy baseline. "
            "Included so the model sees non-pathological trajectories."
        ),
        sessions=[
            _ZONES["content"],
            _ZONES["grateful"],
            _ZONES["content"],
            _ZONES["hopeful"],
            _ZONES["content"],
            _ZONES["calm"],
        ],
        typical_duration_weeks=(4, 12),
        weight=0.7,
    ),

    ArcTemplate(
        name="stable_negative",
        description=(
            "Persistent low-grade negative affect without acute crisis or recovery. "
            "Dysthymia-like pattern."
        ),
        sessions=[
            _ZONES["sad"],
            _ZONES["lonely"],
            _ZONES["sad"],
            _ZONES["exhausted"],
            _ZONES["sad"],
            _ZONES["lonely"],
        ],
        typical_duration_weeks=(6, 20),
        weight=0.9,
    ),

    # ------------------------------------------------------------------
    # Arousal-primary arcs — valence stays roughly constant while arousal
    # changes substantially. These are critical for training the model to
    # represent arousal as an independent axis rather than a valence proxy.
    # Without these, next-session prediction is solvable by tracking valence
    # alone, and the arousal dimension never receives gradient pressure.
    # ------------------------------------------------------------------

    ArcTemplate(
        name="excitement_to_contentment",
        description=(
            "High positive valence throughout; arousal decreases as initial "
            "excitement settles into calm contentment. Models the post-achievement "
            "or post-event deactivation phase — e.g. after a new job, relationship, "
            "or goal is reached and the buzz fades into satisfaction. "
            "Pure arousal arc in positive valence territory (Q1 → Q2)."
        ),
        sessions=[
            _ZONES["excited"],    # (+0.6, +0.6)
            _ZONES["excited"],
            _ZONES["hopeful"],    # (+0.5, +0.3) — arousal dropping
            _ZONES["hopeful"],
            _ZONES["content"],    # (+0.5, -0.3)
            _ZONES["grateful"],   # (+0.6, -0.1)
            _ZONES["calm"],       # (+0.3, -0.5) — fully deactivated
        ],
        typical_duration_weeks=(3, 8),
        weight=1.0,
    ),

    ArcTemplate(
        name="anxiety_to_depression",
        description=(
            "Negative valence throughout; arousal decreases as sustained anxiety "
            "exhausts into low-energy depression. Models the transition from active "
            "distress to depleted withdrawal — common in burnout, chronic stress, "
            "and untreated anxiety disorders. "
            "Pure arousal arc in negative valence territory (Q3 → Q4)."
        ),
        sessions=[
            _ZONES["anxious"],    # (-0.4, +0.5)
            _ZONES["anxious"],
            _ZONES["distressed"], # (-0.6, +0.6) — brief escalation
            _ZONES["anxious"],
            _ZONES["sad"],        # (-0.5, -0.3) — arousal dropping
            _ZONES["depressed"],  # (-0.7, -0.5)
            _ZONES["exhausted"],  # (-0.4, -0.6) — fully depleted
        ],
        typical_duration_weeks=(4, 10),
        weight=1.1,
    ),

    ArcTemplate(
        name="depression_to_anxiety",
        description=(
            "Negative valence throughout; arousal increases as withdrawn depression "
            "activates into anxious rumination or agitation — without positive recovery. "
            "Models re-engagement with distress rather than healing: the person becomes "
            "more activated but not more positive. Common when a stressor re-emerges or "
            "avoidance stops working. "
            "Pure arousal arc in negative valence territory (Q4 → Q3). "
            "All waypoints stay at valence ≤ -0.4 to keep the axes independent — "
            "the prior unsettled(-0.1) waypoint introduced a 0.6 valence swing "
            "correlated with the arousal rise, re-entangling the dimensions."
        ),
        sessions=[
            _ZONES["depressed"],   # (-0.7, -0.5)
            _ZONES["exhausted"],   # (-0.4, -0.6)
            _ZONES["sad"],         # (-0.5, -0.3) — slight arousal uptick
            _ZONES["lonely"],      # (-0.6, -0.2)
            _ZONES["anxious"],     # (-0.4, +0.5) — arousal rises, valence stays negative
            _ZONES["anxious"],     # (-0.4, +0.5)
            _ZONES["distressed"],  # (-0.6, +0.6) — high activation, still negative valence
        ],
        typical_duration_weeks=(4, 10),
        weight=1.1,
    ),
]


def get_template_by_name(name: str) -> ArcTemplate:
    for t in ARC_TEMPLATES:
        if t.name == name:
            return t
    raise KeyError(f"No arc template named '{name}'")


def summarize_templates() -> None:
    """Print a human-readable summary of all templates."""
    print(f"{'Name':<30} {'Sessions':>8}  {'Weeks':>8}  {'Weight':>6}")
    print("-" * 60)
    for t in ARC_TEMPLATES:
        wmin, wmax = t.typical_duration_weeks
        print(
            f"{t.name:<30} {len(t.sessions):>8}  "
            f"{wmin:>3}–{wmax:<3}  {t.weight:>6.1f}"
        )


if __name__ == "__main__":
    summarize_templates()
