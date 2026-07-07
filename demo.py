"""
demo.py — Kokoro interactive demo

Each "session" is a short conversation the user types in the chat box.
Click "Submit Session" to encode it and update the emotional state.
The circumplex plot updates after every session showing the trajectory.

Run:
    python demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── PHASE 1: load ML models BEFORE matplotlib/gradio ──────────────────────
# sentence-transformers and matplotlib conflict if matplotlib is imported
# first — the memory-mapped weights corrupt the Agg renderer.
import torch
import numpy as np

from kokoro.encoder import SessionEncoder
from kokoro.transition import TransitionModel
from training.train_probe import ValenceArousalProbe

# ---------------------------------------------------------------------------
# Load models (once, at startup)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
CKPT_MODEL   = PROJECT_ROOT / "checkpoints" / "transition_v2.pt"
CKPT_PROBE   = PROJECT_ROOT / "checkpoints" / "valence_arousal_probe_v2.pt"

print("Loading transition model…")
from kokoro.transition import load_transition_checkpoint
model, cfg = load_transition_checkpoint(CKPT_MODEL)
use_vad    = cfg.get("session_dim", 384) > 384
state_dim  = cfg.get("state_dim", 384)

print("Loading probe…")
probe_ckpt = torch.load(CKPT_PROBE, map_location="cpu", weights_only=False)
probe      = ValenceArousalProbe(state_dim)
probe.load_state_dict(probe_ckpt["model_state_dict"])
probe.eval()

print("Loading encoder…")
encoder = SessionEncoder(device="cpu", use_vad_features=use_vad)
_ = encoder.model   # trigger download now
print("All models ready.\n")

# ── PHASE 2: now safe to import matplotlib and gradio ─────────────────────
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUADRANTS = {
    "Q1": dict(label="Q1 — Excited / Happy",   color="#2ECC71", v=(0, 1),  a=(0, 1)),
    "Q2": dict(label="Q2 — Calm / Content",     color="#3498DB", v=(0, 1),  a=(-1, 0)),
    "Q3": dict(label="Q3 — Anxious / Stressed", color="#E67E22", v=(-1, 0), a=(0, 1)),
    "Q4": dict(label="Q4 — Sad / Withdrawn",    color="#95A5A6", v=(-1, 0), a=(-1, 0)),
}

def get_quadrant(v: float, a: float) -> str:
    if v >= 0 and a >= 0:  return "Q1"
    if v >= 0 and a <  0:  return "Q2"
    if v <  0 and a >= 0:  return "Q3"
    return "Q4"


# ---------------------------------------------------------------------------
# Core: process one session
# ---------------------------------------------------------------------------
def process_session(chat_history: list[list[str]], state_tensor, trajectory):
    """
    chat_history: list of [user_msg, assistant_msg] pairs
    state_tensor: serialised state as list[float] (None → zero state)
    trajectory:   list of {"v": float, "a": float, "q": str, "session": int}
    """
    if not chat_history:
        return state_tensor, trajectory, None, "No conversation yet."

    # Rebuild turns from chat history (type="messages" format: list of dicts)
    turns = []
    for msg in chat_history:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if content and role in ("user", "assistant"):
            turns.append({"role": role, "content": content})

    if not turns:
        return state_tensor, trajectory, None, "No conversation yet."

    # Encode
    with torch.no_grad():
        emb_np = encoder.encode(turns)              # numpy (session_dim,)
        emb    = torch.tensor(emb_np, dtype=torch.float32)

        # Restore state
        if state_tensor is None:
            state = torch.zeros(state_dim)
        else:
            state = torch.tensor(state_tensor, dtype=torch.float32)

        # Update state
        new_state = model(state, emb)

        # Decode valence/arousal
        va = probe(new_state.unsqueeze(0)).squeeze(0)
        v  = float(va[0].item())
        a  = float(va[1].item())

    q    = get_quadrant(v, a)
    qinfo = QUADRANTS[q]
    sess  = len(trajectory) + 1

    trajectory = trajectory + [{"v": v, "a": a, "q": q, "session": sess}]

    status = (
        f"**Session {sess}** processed\n\n"
        f"**Valence:** {v:+.3f}   |   **Arousal:** {a:+.3f}\n\n"
        f"**State:** {qinfo['label']}"
    )

    fig = draw_circumplex(trajectory)
    return new_state.tolist(), trajectory, fig, status


# ---------------------------------------------------------------------------
# Circumplex plot
# ---------------------------------------------------------------------------
def draw_circumplex(trajectory: list[dict]):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_facecolor("#F8F9FA")
    fig.patch.set_facecolor("#FFFFFF")

    # Quadrant fills
    for q, info in QUADRANTS.items():
        vr, ar = info["v"], info["a"]
        ax.fill_between(
            [vr[0], vr[1]], ar[0], ar[1],
            color=info["color"], alpha=0.08,
        )

    # Axes
    ax.axhline(0, color="#CCCCCC", lw=0.8, zorder=1)
    ax.axvline(0, color="#CCCCCC", lw=0.8, zorder=1)

    # Quadrant labels
    for q, info in QUADRANTS.items():
        cx = 0.55 if info["v"][0] == 0 else -0.55
        cy = 0.82 if info["a"][0] == 0 else -0.82
        ax.text(cx, cy, q, ha="center", va="center",
                fontsize=9, color=info["color"], fontweight="bold", alpha=0.6)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Valence  (negative ← → positive)", fontsize=9)
    ax.set_ylabel("Arousal  (low ← → high)", fontsize=9)
    ax.set_title("Emotional State Trajectory", fontsize=11, fontweight="bold", pad=10)

    if not trajectory:
        plt.tight_layout()
        return fig

    vs = [p["v"] for p in trajectory]
    as_ = [p["a"] for p in trajectory]

    # Draw trajectory path
    for i in range(len(trajectory) - 1):
        ax.annotate(
            "", xy=(vs[i+1], as_[i+1]), xytext=(vs[i], as_[i]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#555566",
                lw=1.4,
                mutation_scale=10,
            ),
            zorder=3,
        )

    # Color gradient: earlier = lighter, later = darker
    n = len(trajectory)
    cmap = plt.get_cmap("Blues")
    for i, pt in enumerate(trajectory):
        alpha = 0.4 + 0.6 * (i / max(n - 1, 1))
        color = cmap(0.35 + 0.55 * (i / max(n - 1, 1)))
        ax.scatter(pt["v"], pt["a"], color=color, s=90, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.text(pt["v"] + 0.04, pt["a"] + 0.04,
                str(pt["session"]),
                fontsize=7.5, color="#333344", zorder=6)

    # Highlight latest point
    ax.scatter(vs[-1], as_[-1], color="#E74C3C", s=130, zorder=6,
               edgecolors="white", linewidths=1.8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def reset_all():
    return None, [], draw_circumplex([]), "State reset. Start a new conversation."


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
INTRO = """
## Kokoro — Affective World Model Demo

**How to use:**
1. Have a conversation in the chat box below (type your message, press Enter, add the AI's reply)
2. Click **Submit Session** when the conversation is complete
3. Watch the circumplex plot update with the predicted emotional state
4. Start a new conversation for the next session — the state carries over

The model maintains a 384-dimensional latent state that evolves across sessions.
A linear probe decodes valence and arousal; a decoder maps those to Q1–Q4.
"""

with gr.Blocks(title="Kokoro Demo", theme=gr.themes.Soft()) as demo:
    # Persistent state (per browser session)
    state_vec  = gr.State(None)        # list[float] or None
    trajectory = gr.State([])          # list of {v, a, q, session}

    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation (this session)",
                height=380,
                type="messages",
            )
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Type a message…",
                    label="",
                    scale=5,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", scale=1, variant="secondary")

            with gr.Row():
                submit_btn = gr.Button("✓  Submit Session", variant="primary", scale=3)
                reset_btn  = gr.Button("↺  Reset All", variant="stop", scale=1)

            status_box = gr.Markdown("_No sessions processed yet._")

        with gr.Column(scale=2):
            plot = gr.Plot(label="Emotional State Circumplex", show_label=True)

    # ---- event wiring ----

    def add_user_message(msg, history):
        if msg.strip():
            return "", history + [{"role": "user", "content": msg}]
        return msg, history

    def add_assistant_reply(history):
        # Just move focus — user fills in the assistant turn manually
        # Or if last turn has no assistant reply, prompt for it
        return history

    # Send user message
    send_btn.click(
        fn=add_user_message,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    )
    user_input.submit(
        fn=add_user_message,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot],
    )

    # Submit session
    submit_btn.click(
        fn=process_session,
        inputs=[chatbot, state_vec, trajectory],
        outputs=[state_vec, trajectory, plot, status_box],
    ).then(
        fn=lambda: [],     # clear chat for next session
        outputs=[chatbot],
    )

    # Reset
    reset_btn.click(
        fn=reset_all,
        outputs=[state_vec, trajectory, plot, status_box],
    ).then(
        fn=lambda: [],
        outputs=[chatbot],
    )

    # Initial plot
    demo.load(fn=lambda: draw_circumplex([]), outputs=[plot])

    gr.Markdown("""
---
**Model:** Transition Model (split LayerNorm + VICReg + VAD) · PR = 339.7 / 384
**Probe:** Linear(384→2) · valence r = 0.698 · arousal r = 0.542
**Training data:** 10,000 synthetic trajectories across 14 emotional arc types
""")


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
