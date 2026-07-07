"""
data/low_arousal_pool.py — synthetic deep-low-arousal session pool.

Why this exists
---------------
EmpatheticDialogues imposes a hard arousal floor of about -0.30 on the training
data: its lowest-arousal labels (sad -0.25, devastated -0.30, content -0.25)
never reach the deep-deactivation region of the circumplex. Arc templates that
specify waypoints like exhausted(-0.4, -0.6) or depressed(-0.7, -0.5) silently
fall back to ~-0.25-arousal conversations, so the model has NEVER seen a
genuinely low-energy session paired with a genuinely low arousal label. This is
the single documented cause of the arousal probe ceiling (r = 0.542 vs 0.698
for valence) — see research_report.md §13.1.

This module generates templated first-person sessions that live specifically in
the missing region:

  Q4-deep  (negative, deactivated):  v ∈ [-0.85, -0.35], a ∈ [-0.80, -0.35]
           exhaustion, numbness, depressive shutdown, anhedonia
  Q2-deep  (positive, deactivated):  v ∈ [+0.30, +0.75], a ∈ [-0.80, -0.40]
           deep calm, serenity, restful contentment

Both regions are needed: with only Q4-deep, "low arousal" would become a proxy
for "negative valence" and re-entangle the axes the arousal-primary arcs were
built to separate.

The texts are template-composed (subject fragments × state fragments ×
elaborations), producing thousands of distinct surface forms. They are not as
rich as human data — this is explicitly the best available move WITHIN the
current data constraints, and the honest fix remains a real low-arousal corpus
(see research_report.md §15).

Usage:
    from data.low_arousal_pool import load_low_arousal_sessions
    pool += load_low_arousal_sessions(n=600, seed=42)

or via the trajectory builder:
    python -m data.construct_trajectories 10000 --augment-low-arousal
"""

from __future__ import annotations

import random

from data.prepare import LabeledSession

# ---------------------------------------------------------------------------
# Template fragments — Q4-deep (negative valence, deep deactivation)
# ---------------------------------------------------------------------------

_Q4_OPENERS = [
    "i don't really have the energy to talk much today",
    "everything feels heavy. even typing this",
    "i slept eleven hours and i'm still exhausted",
    "i've been lying on the couch all day. can't make myself move",
    "i feel completely drained. empty tank",
    "another day where i did basically nothing. no energy at all",
    "i feel numb. not sad exactly. just nothing",
    "can't remember the last time i felt awake",
    "i cancelled on my friends again. couldn't face getting dressed",
    "food doesn't taste like anything lately. i eat because i have to",
    "i stared at the wall for an hour this morning. didn't even notice time passing",
    "my limbs feel like they're full of sand",
]

_Q4_MIDDLES = [
    "it's not panic or anything. there's no racing thoughts. just this flatness",
    "i'm not crying. i'm not anxious. i just feel switched off",
    "nothing sounds interesting. not tv, not music, not anything",
    "i used to care about things. right now i can't find the caring",
    "getting out of bed took everything i had and that was the whole day",
    "people ask what's wrong and i genuinely don't have an answer. i'm just empty",
    "i'm so tired all the time. sleep doesn't fix it",
    "showering feels like a project. i keep putting it off",
    "the days blur. i couldn't tell you what i did yesterday",
    "it's like the volume on everything got turned down to almost zero",
]

_Q4_CLOSERS = [
    "i just want to sleep and not have to be anything for a while",
    "maybe tomorrow will have more in it. today is just this",
    "i'm not in danger or anything. just very very tired of everything",
    "sorry i'm not more talkative. this is all i've got today",
    "i'll try to eat something proper tonight. that's the whole plan",
    "thanks for being here. i don't have much to say but it helps",
]

_Q4_ASSISTANT = [
    "You don't have to perform energy you don't have. I'm here either way.",
    "That deep flatness is exhausting in its own way. Thank you for telling me.",
    "It's okay for today to just be about getting through. No pressure here.",
    "Low days like this are heavy. You showed up anyway, and that counts.",
]

# ---------------------------------------------------------------------------
# Template fragments — Q2-deep (positive valence, deep deactivation)
# ---------------------------------------------------------------------------

_Q2_OPENERS = [
    "spent the whole afternoon reading by the window. so peaceful",
    "took a long slow walk by the river today. completely unhurried",
    "had a quiet weekend at the cabin. no phone, no plans",
    "sat in the garden with tea this morning and just watched the birds",
    "did absolutely nothing today and it was wonderful",
    "long bath, early night, no obligations. exactly what i needed",
    "the house is quiet, the rain is soft, and i have nowhere to be",
    "meditated for a while this evening. everything feels settled",
]

_Q2_MIDDLES = [
    "no rush, no noise. just calm. i can feel my shoulders unclenching",
    "i feel rested in a way i haven't in months. deeply, properly rested",
    "there's this quiet contentment, like everything is where it should be",
    "i wasn't excited or anything. just... at ease. still water",
    "my mind is quiet for once. no to-do list running in the background",
    "it's the kind of slow day that makes you feel human again",
]

_Q2_CLOSERS = [
    "going to make some soup and turn in early. lovely end to a lovely day",
    "i want more days like this. slow and soft",
    "nothing to report really. just peace. wanted to share it",
    "feeling very grateful for the quiet",
]

_Q2_ASSISTANT = [
    "That sounds genuinely restorative. Days like that recharge something deep.",
    "What a peaceful picture. I'm glad you gave yourself that stillness.",
    "That quiet kind of contentment is precious. Savor it.",
]


def _make_session(
    rng: random.Random,
    openers: list[str],
    middles: list[str],
    closers: list[str],
    assistant: list[str],
) -> list[dict[str, str]]:
    turns = [
        {"role": "user",      "content": rng.choice(openers)},
        {"role": "assistant", "content": rng.choice(assistant)},
        {"role": "user",      "content": rng.choice(middles)},
    ]
    if rng.random() < 0.7:
        turns.append({"role": "assistant", "content": rng.choice(assistant)})
        turns.append({"role": "user",      "content": rng.choice(closers)})
    return turns


def load_low_arousal_sessions(
    n: int = 600,
    seed: int = 42,
    q4_fraction: float = 0.65,
) -> list[LabeledSession]:
    """Generate a pool of deep-low-arousal LabeledSessions.

    Parameters
    ----------
    n:           Total sessions to generate.
    seed:        RNG seed (texts and coordinate jitter are reproducible).
    q4_fraction: Fraction placed in the negative-valence Q4-deep region;
                 the rest go to the positive-valence Q2-deep region.

    Returns
    -------
    LabeledSessions with conv_ids "synthlow_q4_<i>" / "synthlow_q2_<i>" and
    ground-truth (valence, arousal) sampled uniformly inside the target
    region — crucially with arousal BELOW the EmpatheticDialogues floor.
    """
    rng = random.Random(seed)
    sessions: list[LabeledSession] = []
    n_q4 = int(n * q4_fraction)

    for i in range(n_q4):
        sessions.append(LabeledSession(
            conv_id=f"synthlow_q4_{i}",
            turns=_make_session(rng, _Q4_OPENERS, _Q4_MIDDLES, _Q4_CLOSERS, _Q4_ASSISTANT),
            valence=rng.uniform(-0.85, -0.35),
            arousal=rng.uniform(-0.80, -0.35),
            emotion_label="synth_depleted",
        ))

    for i in range(n - n_q4):
        sessions.append(LabeledSession(
            conv_id=f"synthlow_q2_{i}",
            turns=_make_session(rng, _Q2_OPENERS, _Q2_MIDDLES, _Q2_CLOSERS, _Q2_ASSISTANT),
            valence=rng.uniform(+0.30, +0.75),
            arousal=rng.uniform(-0.80, -0.40),
            emotion_label="synth_serene",
        ))

    return sessions


if __name__ == "__main__":
    pool = load_low_arousal_sessions(n=20)
    for s in pool[:4] + pool[-2:]:
        print(f"[{s.conv_id}] v={s.valence:+.2f} a={s.arousal:+.2f} ({s.emotion_label})")
        for t in s.turns[:2]:
            print(f"  {t['role'][:4]}: {t['content'][:80]}")
        print()
    from collections import Counter
    print(Counter(s.emotion_label for s in pool))
