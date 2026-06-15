"""
Probe generalization test on naturalistic companion AI conversations.

The linear probe was trained exclusively on EmpatheticDialogues-style synthetic
text. This test checks whether the probe's predictions are meaningful when
presented with naturalistic companion AI conversations — the actual deployment
context — which look nothing like EmpatheticDialogues.

Protocol:
  1. 24 handcrafted naturalistic conversation scenarios
  2. Each covers a distinct emotional region in the Russell circumplex
  3. Run each through: SessionEncoder -> TransitionModel -> LinearProbe
  4. Report predicted (valence, arousal) and expected quadrant
  5. Compute output range — key indicator of generalization vs collapse

Original paper finding (PR=1.4 model):
  Probe output range on naturalistic text: -0.144 to -0.110 (valence)
  i.e. nearly identical output regardless of emotional content

Expected on current model (PR=339.7):
  If generalization improved: wider range, directionally correct predictions
  If same failure: narrow range (< 0.05 spread), direction misses

Run:
    python -m diagnostics.probe_generalization
    python -m diagnostics.probe_generalization --checkpoint checkpoints/transition_v1.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger(__name__)

W = 76


def hline(c: str = "-") -> None:
    print(c * W)


# ---------------------------------------------------------------------------
# Naturalistic scenarios
# Each scenario: list of {"role": "user"|"assistant", "content": str}
# Expected quadrant for reference (not used in computation)
# ---------------------------------------------------------------------------

SCENARIOS: list[dict] = [
    # -----------------------------------------------------------------------
    # Quadrant 1: High valence, High arousal (excited, enthusiastic)
    # -----------------------------------------------------------------------
    {
        "label": "Q1-job_offer",
        "expected_valence": "positive",
        "expected_arousal": "high",
        "expected_quadrant": "Q1 (+v, +a)",
        "turns": [
            {"role": "user",      "content": "I got the job!! They just called me and I got the offer! I can't believe it, I've been applying for months and this is my dream company. I'm literally shaking right now!"},
            {"role": "assistant", "content": "That's incredible news! Congratulations, you've worked so hard for this. How are you feeling?"},
            {"role": "user",      "content": "I feel like I could run a marathon right now. I want to call everyone I know. This changes everything for me, I'm just so pumped and relieved and happy all at once!"},
        ],
    },
    {
        "label": "Q1-wedding",
        "expected_valence": "positive",
        "expected_arousal": "high",
        "expected_quadrant": "Q1 (+v, +a)",
        "turns": [
            {"role": "user",      "content": "She said yes!! I proposed last night and she said yes! I planned everything — the restaurant, the ring — and it went perfectly. I'm getting married!"},
            {"role": "assistant", "content": "Oh my goodness, that's wonderful! You must be over the moon. Tell me everything!"},
            {"role": "user",      "content": "I'm so happy I barely slept last night. We were both crying and laughing and calling family. I can't stop smiling. This is the best day of my life."},
        ],
    },
    {
        "label": "Q1-startup_funding",
        "expected_valence": "positive",
        "expected_arousal": "high",
        "expected_quadrant": "Q1 (+v, +a)",
        "turns": [
            {"role": "user",      "content": "We just closed our Series A! 4 million dollars. Two years of grinding and we finally did it. The team is ecstatic, we're celebrating tonight."},
            {"role": "assistant", "content": "That's a massive milestone. How's the energy in the office right now?"},
            {"role": "user",      "content": "Electric. Everyone's talking over each other, we ordered pizza, someone brought champagne. I feel validated, energized, like we can actually build this thing now."},
        ],
    },

    # -----------------------------------------------------------------------
    # Quadrant 2: Low valence, High arousal (anxious, panicked, angry)
    # -----------------------------------------------------------------------
    {
        "label": "Q2-panic_attack",
        "expected_valence": "negative",
        "expected_arousal": "high",
        "expected_quadrant": "Q2 (-v, +a)",
        "turns": [
            {"role": "user",      "content": "I had a panic attack at work today. Heart pounding, couldn't breathe, had to leave the meeting. I'm so embarrassed and scared it's going to happen again."},
            {"role": "assistant", "content": "That sounds really frightening. Are you somewhere safe right now?"},
            {"role": "user",      "content": "Yeah I'm home but I keep thinking about it. My hands are still a bit shaky. I can't stop worrying that everyone at work noticed and thinks I'm unstable."},
        ],
    },
    {
        "label": "Q2-relationship_fight",
        "expected_valence": "negative",
        "expected_arousal": "high",
        "expected_quadrant": "Q2 (-v, +a)",
        "turns": [
            {"role": "user",      "content": "We had the biggest fight of our relationship last night. I'm furious and hurt. He said things you can't take back. I'm pacing around the apartment and I can't calm down."},
            {"role": "assistant", "content": "That sounds really intense. What happened?"},
            {"role": "user",      "content": "He basically said I always make everything about myself. Out of nowhere. I'm so angry I can't think straight. Part of me wants to end it, part of me wants to scream. I don't know what to do with all this energy."},
        ],
    },
    {
        "label": "Q2-deadline_anxiety",
        "expected_valence": "negative",
        "expected_arousal": "high",
        "expected_quadrant": "Q2 (-v, +a)",
        "turns": [
            {"role": "user",      "content": "I have a presentation to 50 executives in 3 hours and I just realized there's a major error in my data. My heart is racing, I feel sick."},
            {"role": "assistant", "content": "OK, let's breathe first. Is it fixable in 3 hours?"},
            {"role": "user",      "content": "Maybe? I don't know. I'm spiraling. I can't afford to mess this up, this project is tied to my performance review. I'm terrified of looking incompetent in front of them."},
        ],
    },

    # -----------------------------------------------------------------------
    # Quadrant 3: Low valence, Low arousal (depressed, hopeless, empty)
    # -----------------------------------------------------------------------
    {
        "label": "Q3-depression_episode",
        "expected_valence": "negative",
        "expected_arousal": "low",
        "expected_quadrant": "Q3 (-v, -a)",
        "turns": [
            {"role": "user",      "content": "I didn't get out of bed until 2pm again. I don't really see the point of anything lately. I'm not sad exactly, just... empty. Everything feels like too much effort."},
            {"role": "assistant", "content": "I hear you. That heaviness sounds really hard to carry. How long has it felt this way?"},
            {"role": "user",      "content": "Few weeks now. I'm barely keeping up with work. I haven't called my friends back. I don't know what's wrong with me, I just have no energy and no motivation and I feel kind of numb."},
        ],
    },
    {
        "label": "Q3-grief",
        "expected_valence": "negative",
        "expected_arousal": "low",
        "expected_quadrant": "Q3 (-v, -a)",
        "turns": [
            {"role": "user",      "content": "My dad passed away last month. I thought I'd be doing better by now but I'm still just... sitting with it. I don't cry as much anymore. Mostly I just feel this dull weight everywhere."},
            {"role": "assistant", "content": "Grief doesn't really follow a timeline. How are you taking care of yourself?"},
            {"role": "user",      "content": "Not great, honestly. I eat okay I guess. I go through the motions at work. But there's this quiet sadness underneath everything. Like a background hum. I miss him so much but I don't even have the energy to cry."},
        ],
    },
    {
        "label": "Q3-burnout",
        "expected_valence": "negative",
        "expected_arousal": "low",
        "expected_quadrant": "Q3 (-v, -a)",
        "turns": [
            {"role": "user",      "content": "I think I'm completely burned out. I used to love my work and now I dread opening my laptop. I'm exhausted all the time even when I sleep. Nothing is fun anymore, not even things I used to enjoy."},
            {"role": "assistant", "content": "Burnout can hollow things out like that. How long have you been running on empty?"},
            {"role": "user",      "content": "Six months at least. I just feel flat. My colleagues seem energetic and I just wonder where that went for me. I'm tired of being tired. I don't even have the energy to figure out how to fix it."},
        ],
    },

    # -----------------------------------------------------------------------
    # Quadrant 4: High valence, Low arousal (calm, content, peaceful)
    # -----------------------------------------------------------------------
    {
        "label": "Q4-vacation_peace",
        "expected_valence": "positive",
        "expected_arousal": "low",
        "expected_quadrant": "Q4 (+v, -a)",
        "turns": [
            {"role": "user",      "content": "I'm on vacation in Portugal and I'm just sitting at this little cafe watching the street. No agenda, no phone calls, nowhere to be. It's really nice."},
            {"role": "assistant", "content": "That sounds wonderful. What's it like to have that kind of stillness?"},
            {"role": "user",      "content": "Honestly? Really good. I keep waiting to feel restless but I don't. I feel settled. Happy to just exist here and drink good coffee. Simple but it's exactly what I needed."},
        ],
    },
    {
        "label": "Q4-after_therapy_breakthrough",
        "expected_valence": "positive",
        "expected_arousal": "low",
        "expected_quadrant": "Q4 (+v, -a)",
        "turns": [
            {"role": "user",      "content": "I had a really good therapy session today. We worked through something that's been weighing on me for years. I feel lighter, like I've put something down that I didn't realize I was carrying."},
            {"role": "assistant", "content": "That kind of shift in therapy can be really meaningful. What are you noticing now?"},
            {"role": "user",      "content": "A kind of quiet. Not bored quiet, more like settled. Content, I think. I'm going to take a slow walk home and just sit with it. I feel okay."},
        ],
    },
    {
        "label": "Q4-new_routine_satisfaction",
        "expected_valence": "positive",
        "expected_arousal": "low",
        "expected_quadrant": "Q4 (+v, -a)",
        "turns": [
            {"role": "user",      "content": "I've been doing the morning walk for three weeks straight now and I love it. Nothing dramatic, just 30 minutes before work. It's become my favorite part of the day."},
            {"role": "assistant", "content": "There's something really solid about a simple routine like that. What do you like about it?"},
            {"role": "user",      "content": "It's just mine. No emails, no decisions. Just me and the neighborhood. I come back calm and ready. It's quiet in a good way. I feel genuinely well."},
        ],
    },

    # -----------------------------------------------------------------------
    # Transition / mixed states
    # -----------------------------------------------------------------------
    {
        "label": "TRANS-recovery_starting",
        "expected_valence": "slightly positive",
        "expected_arousal": "moderate",
        "expected_quadrant": "recovering (Q3 -> Q4)",
        "turns": [
            {"role": "user",      "content": "I've been in a rough patch the last few months but I think I'm starting to come out of it. I actually wanted to cook dinner today, which I haven't cared about in a long time."},
            {"role": "assistant", "content": "That sounds like a meaningful shift. What else is different?"},
            {"role": "user",      "content": "Small things. I texted a friend back. I took a shower without it being a big deal. It's not like I'm happy, but I don't feel completely underwater anymore."},
        ],
    },
    {
        "label": "TRANS-stress_stabilizing",
        "expected_valence": "neutral",
        "expected_arousal": "moderate",
        "expected_quadrant": "stabilizing (Q2 -> neutral)",
        "turns": [
            {"role": "user",      "content": "The worst of the work crisis is over. Still a lot to do but the immediate fire is out. I'm tired but not panicking anymore. Starting to feel more like myself."},
            {"role": "assistant", "content": "Coming down from that kind of sustained stress takes time. How are you sleeping?"},
            {"role": "user",      "content": "Better. Last week I was waking up at 3am with my brain running. Now it's just normal tired. I'm not out of the woods but I feel more grounded."},
        ],
    },
    {
        "label": "TRANS-excitement_settling",
        "expected_valence": "positive",
        "expected_arousal": "moderate",
        "expected_quadrant": "settling (Q1 -> Q4)",
        "turns": [
            {"role": "user",      "content": "The new job high is wearing off and I'm settling into the routine. It's good! But not the rush it was. Just normal good, you know?"},
            {"role": "assistant", "content": "That transition from excitement to contentment is its own kind of stability. How does it feel?"},
            {"role": "user",      "content": "Comfortable. I like my team, the work is interesting, I sleep well. Nothing to report really, which after the past year feels like a gift."},
        ],
    },

    # -----------------------------------------------------------------------
    # Ambiguous / low-signal states (hard cases)
    # -----------------------------------------------------------------------
    {
        "label": "AMB-ambiguous_check_in",
        "expected_valence": "neutral",
        "expected_arousal": "low",
        "expected_quadrant": "neutral/flat",
        "turns": [
            {"role": "user",      "content": "hey"},
            {"role": "assistant", "content": "Hey! How are you doing today?"},
            {"role": "user",      "content": "ok i guess. not much going on. just bored"},
        ],
    },
    {
        "label": "AMB-work_fine",
        "expected_valence": "slightly positive",
        "expected_arousal": "low",
        "expected_quadrant": "neutral/slightly positive",
        "turns": [
            {"role": "user",      "content": "Work was fine today. Had a few meetings, got some stuff done. Nothing bad happened. Kind of just a normal day."},
            {"role": "assistant", "content": "Sounds like a solid if unremarkable day. Anything you're thinking about?"},
            {"role": "user",      "content": "Not really. I'm going to cook something and watch TV. I'm okay."},
        ],
    },
    {
        "label": "AMB-cant_sleep",
        "expected_valence": "slightly negative",
        "expected_arousal": "moderate",
        "expected_quadrant": "Q2/Q3 border",
        "turns": [
            {"role": "user",      "content": "Can't sleep. Brain won't stop. Nothing specific, just this low hum of worry about things that probably don't matter."},
            {"role": "assistant", "content": "That restless 3am feeling is its own kind of exhausting. What's looping through your head?"},
            {"role": "user",      "content": "Just like... did I handle that email right, do people at work think I'm competent, will I regret decisions I made years ago. The usual spiral. I'm not in crisis or anything, just annoyed at my brain."},
        ],
    },

    # -----------------------------------------------------------------------
    # High-signal explicit scenarios
    # -----------------------------------------------------------------------
    {
        "label": "Q1-reconnection",
        "expected_valence": "positive",
        "expected_arousal": "high",
        "expected_quadrant": "Q1 (+v, +a)",
        "turns": [
            {"role": "user",      "content": "I ran into my best friend from college today, completely randomly. We haven't talked in 5 years. We ended up talking for 3 hours at a coffee shop. It was like no time had passed!"},
            {"role": "assistant", "content": "That kind of reconnection is rare and special. How did it feel?"},
            {"role": "user",      "content": "Like a jolt of pure joy. We were laughing and catching up and it reminded me of who I used to be. I feel alive in a way I haven't in a while. I came home buzzing."},
        ],
    },
    {
        "label": "Q3-isolation",
        "expected_valence": "negative",
        "expected_arousal": "low",
        "expected_quadrant": "Q3 (-v, -a)",
        "turns": [
            {"role": "user",      "content": "I haven't left the apartment in 4 days. I know I should go outside but I just can't seem to make myself do it. Everything feels pointless and I don't see why it matters."},
            {"role": "assistant", "content": "Four days is a long time. What's been stopping you?"},
            {"role": "user",      "content": "I don't know. It's not that I'm scared or anxious, I'm just... flat. Like the volume got turned down on everything. I do the minimum to get through the day and then it's over."},
        ],
    },
    {
        "label": "Q2-anticipatory_dread",
        "expected_valence": "negative",
        "expected_arousal": "high",
        "expected_quadrant": "Q2 (-v, +a)",
        "turns": [
            {"role": "user",      "content": "I have a really difficult conversation I need to have with my brother this weekend and I've been dreading it all week. I can't stop thinking about how it might go wrong."},
            {"role": "assistant", "content": "Anticipatory anxiety before hard conversations can be as draining as the conversation itself. What are you most worried about?"},
            {"role": "user",      "content": "That he'll get defensive and we'll say things we regret and it'll damage the relationship. My stomach is in knots. I keep rehearsing it in my head but that makes it worse."},
        ],
    },
    {
        "label": "Q4-milestone_gratitude",
        "expected_valence": "positive",
        "expected_arousal": "low",
        "expected_quadrant": "Q4 (+v, -a)",
        "turns": [
            {"role": "user",      "content": "I turned 40 today. I thought I'd feel something dramatic but mostly I just feel grateful. A lot has happened, good and bad, and I'm glad I'm still here."},
            {"role": "assistant", "content": "That's a reflective way to mark the day. What are you most grateful for?"},
            {"role": "user",      "content": "My health. My people. The fact that I know myself better than I did at 30. I feel settled in a way I didn't expect. Quiet and okay. Not bad for 40."},
        ],
    },
]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_probe_on_scenario(
    scenario: dict,
    encoder,
    model,
    probe,
    device: torch.device,
) -> dict:
    """Run a single scenario through encoder → transition model → probe."""
    import torch

    turns = scenario["turns"]

    # Encode
    session_emb = encoder.encode(turns)                              # np.ndarray
    emb_t = torch.tensor(session_emb, dtype=torch.float32, device=device)

    # Single-step transition from zero state (new user, first session)
    state = torch.zeros(model.STATE_DIM, device=device)
    with torch.no_grad():
        state = model(state, emb_t)
        va = probe(state.unsqueeze(0)).squeeze(0)                    # (2,)
    valence = va[0].item()
    arousal = va[1].item()

    return {
        "label":             scenario["label"],
        "expected_valence":  scenario["expected_valence"],
        "expected_arousal":  scenario["expected_arousal"],
        "expected_quadrant": scenario["expected_quadrant"],
        "predicted_valence": valence,
        "predicted_arousal": arousal,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def direction_match(predicted: float, expected_label: str) -> str:
    """Check if prediction direction matches expected label."""
    positive = expected_label in ("positive", "slightly positive")
    negative = expected_label in ("negative", "slightly negative")
    if positive and predicted > 0.05:
        return "correct"
    if negative and predicted < -0.05:
        return "correct"
    if expected_label in ("neutral", "moderate", "slightly positive", "slightly negative"):
        return "~ok"
    return "miss"


def print_report(results: list[dict]) -> None:
    vals    = [r["predicted_valence"] for r in results]
    arousals= [r["predicted_arousal"] for r in results]

    v_range = max(vals) - min(vals)
    a_range = max(arousals) - min(arousals)

    print()
    hline("=")
    print("  Probe Generalization Test — Naturalistic Companion AI Conversations")
    hline("=")
    print(f"  Scenarios:  {len(results)}")
    print(f"  Valence range:  [{min(vals):.3f}, {max(vals):.3f}]  (spread = {v_range:.3f})")
    print(f"  Arousal range:  [{min(arousals):.3f}, {max(arousals):.3f}]  (spread = {a_range:.3f})")
    print()

    # Interpretation of range
    print("  Generalization assessment:")
    if v_range < 0.05:
        print(f"  VALENCE: COLLAPSED (range {v_range:.3f}) — same as original paper failure")
    elif v_range < 0.20:
        print(f"  VALENCE: PARTIAL ({v_range:.3f} range) — some sensitivity but limited")
    else:
        print(f"  VALENCE: GENERALIZING ({v_range:.3f} range) — meaningfully differentiated")

    if a_range < 0.05:
        print(f"  AROUSAL: COLLAPSED (range {a_range:.3f})")
    elif a_range < 0.20:
        print(f"  AROUSAL: PARTIAL ({a_range:.3f} range)")
    else:
        print(f"  AROUSAL: GENERALIZING ({a_range:.3f} range)")

    print()
    print(f"  Paper reference (PR=1.4 model): valence range -0.144 to -0.110 (spread=0.034)")
    hline()

    # Per-scenario table
    print()
    hline("=")
    print("  Per-scenario predictions")
    hline("=")
    print(f"  {'Scenario':<28}  {'Exp.Quad':<22}  {'Pred.V':>7}  {'Pred.A':>7}  {'V-dir':>7}")
    hline()

    for r in results:
        vm = direction_match(r["predicted_valence"], r["expected_valence"])
        print(
            f"  {r['label']:<28}  {r['expected_quadrant']:<22}  "
            f"{r['predicted_valence']:>7.3f}  {r['predicted_arousal']:>7.3f}  "
            f"{vm:>7}"
        )
    hline("=")

    # Summary by quadrant
    q1 = [r for r in results if r["expected_quadrant"].startswith("Q1")]
    q2 = [r for r in results if r["expected_quadrant"].startswith("Q2")]
    q3 = [r for r in results if r["expected_quadrant"].startswith("Q3")]
    q4 = [r for r in results if r["expected_quadrant"].startswith("Q4")]

    print()
    print("  Quadrant mean predictions:")
    hline()
    for qname, qlist in [("Q1 (+v, +a)", q1), ("Q2 (-v, +a)", q2),
                          ("Q3 (-v, -a)", q3), ("Q4 (+v, -a)", q4)]:
        if not qlist:
            continue
        mv = np.mean([r["predicted_valence"] for r in qlist])
        ma = np.mean([r["predicted_arousal"] for r in qlist])
        print(f"  {qname:<18}  mean_valence={mv:+.3f}  mean_arousal={ma:+.3f}  (N={len(qlist)})")
    hline()

    # Directionality check: Q1 mean_v > Q3 mean_v?
    if q1 and q3:
        q1_v = np.mean([r["predicted_valence"] for r in q1])
        q3_v = np.mean([r["predicted_valence"] for r in q3])
        print(f"\n  Valence ordering check (Q1 > Q3): {q1_v:.3f} vs {q3_v:.3f} -> {'PASS' if q1_v > q3_v else 'FAIL'}")
    if q2 and q4:
        q2_a = np.mean([r["predicted_arousal"] for r in q2])
        q4_a = np.mean([r["predicted_arousal"] for r in q4])
        print(f"  Arousal ordering check (Q2 > Q4): {q2_a:.3f} vs {q4_a:.3f} -> {'PASS' if q2_a > q4_a else 'FAIL'}")
    hline("=")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Probe generalization on naturalistic text")
    parser.add_argument("--checkpoint",        type=str, default="checkpoints/transition_v1.pt")
    parser.add_argument("--probe-checkpoint",  type=str, default="checkpoints/valence_arousal_probe.pt")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    ckpt_path  = root / args.checkpoint
    probe_path = root / args.probe_checkpoint

    for p in [ckpt_path, probe_path]:
        if not p.exists():
            sys.exit(f"Not found: {p}")

    device = torch.device("cpu")

    # Load transition model
    logger.info("Loading transition model...")
    from kokoro.transition import TransitionModel
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("model_config", {})
    model = TransitionModel(**{k: v for k, v in cfg.items() if k in ("state_dim", "session_dim", "hidden_dim")})
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    use_vad = cfg.get("session_dim", 384) > 384
    logger.info(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, use_vad={use_vad}")

    # Load probe
    logger.info("Loading linear probe...")
    from training.train_probe import ValenceArousalProbe
    probe_ckpt = torch.load(probe_path, map_location="cpu", weights_only=False)
    probe_cfg  = probe_ckpt.get("probe_config", {})
    probe = ValenceArousalProbe(state_dim=probe_cfg.get("state_dim", 384))
    probe.load_state_dict(probe_ckpt["model_state_dict"])
    probe.eval()
    logger.info(f"  val_r_valence={probe_ckpt.get('val_r_valence', '?'):.4f}  "
                f"val_r_arousal={probe_ckpt.get('val_r_arousal', '?'):.4f}")

    # Load encoder
    logger.info("Loading encoder...")
    from kokoro.encoder import SessionEncoder
    encoder = SessionEncoder(device="cpu", use_vad_features=use_vad)
    _ = encoder.model  # warm up

    # Run all scenarios
    logger.info(f"Running {len(SCENARIOS)} naturalistic scenarios...")
    results = []
    for scenario in SCENARIOS:
        r = run_probe_on_scenario(scenario, encoder, model, probe, device)
        results.append(r)
        logger.debug(f"  {r['label']}: v={r['predicted_valence']:.3f}  a={r['predicted_arousal']:.3f}")

    print_report(results)
