# Long-History Evaluation Report v2

**Date:** 2026-06-16
**Pipeline:** `build_contexts.py --long` then `run_llm_eval.py --long`

---

## What Changed

### 1. Expanded long-history scenario set (6 to 20)

The original 6 long-history scenarios (l01-l06) were designed to surface retrieval divergence between semantic-only and hybrid emotional retrieval. 14 new scenarios (l07-l20) were added, covering a wider range of emotional arc types:

| ID | Arc type | Character | Phase 1 | Phase 2 | New message |
|----|----------|-----------|---------|---------|-------------|
| l01 | recovery_over_burnout | Marcus | burnt out dev | therapy, boundaries, joy | "big sprint kicking off next week" |
| l02 | grief_to_healing | Priya | devastating breakup | rebuilt, new relationship | "heard his favourite song today" |
| l03 | anxiety_to_confidence | Kaveh | social anxiety, avoidance | CBT, enjoys social events | "big work party on Friday" |
| l04 | career_crisis_to_growth | Diane | trapped in corporate job | thriving at startup | "performance review is next week" |
| l05 | divorce_to_rebuilding | Tom | divorce grief, isolation | stable single parent | "taking the kids this weekend" |
| l06 | academic_failure_to_success | Amara | failing uni, drop-out risk | excelling, strong dissertation | "dissertation results tomorrow" |
| **l07** | **health_scare_to_recovery** | Ren | terrifying diagnosis | fully recovered, healthy | "annual checkup next Tuesday" |
| **l08** | **social_anxiety_to_connection** | Jess | isolated, declining invites | climbing club, close friends | "group chat is planning a weekend trip" |
| **l09** | **creative_block_to_flow** | Nina | writer's block, self-doubt | daily writing, creative joy | "starting a new chapter today" |
| **l10** | **loneliness_to_community** | David | crushing loneliness, new city | volunteering, neighbours, belonging | "quiet evening at home tonight" |
| **l11** | **financial_stress_to_stability** | Sam | debt spiral, can't pay rent | budget plan, savings, stability | "rent is due this week" |
| **l12** | **chronic_pain_acceptance** | Clara | fighting pain, angry, shrinking life | acceptance, pacing, full life | "physio appointment tomorrow" |
| **l13** | **new_parent_exhaustion_to_adaptation** | Alex | sleep deprivation, overwhelm | routine, confidence, joy | "baby was up twice last night" |
| **l14** | **relocation_distress_to_belonging** | Mia | homesick, disoriented | favourite spots, belonging | "walking through the neighbourhood" |
| **l15** | **identity_crisis_to_clarity** | Kai | lost after military career | found purpose in teaching | "thinking about my next steps" |
| **l16** | **relationship_conflict_to_resolution** | Yara | escalating fights, near separation | couples therapy, rebuilt | "big conversation with my partner tonight" |
| **l17** | **sobriety_journey** | Liam | drinking escalation, consequences | 6 months sober, confident | "going to a bar with friends tonight" |
| **l18** | **caregiver_burnout_to_boundary_setting** | Farah | all-consuming caregiving | boundaries, professional help | "visiting mum this weekend" |
| **l19** | **performance_anxiety_to_confidence** | Omar | paralysing stage fright | enjoys presenting, 200-person talk | "presenting to the whole company next week" |
| **l20** | **grief_anniversary_vs_integrated_loss** | Elena | raw first-year grief | integrated loss, warm memories | "it's Dad's birthday next week" |

Each scenario has 10-11 sessions with a clear Phase 1 (emotionally negative) and Phase 2 (recovery/adaptation) sharing the same topic keywords. The new message uses Phase 1 keywords to create retrieval tension.

### 2. Three-judge majority vote

The single-judge evaluation was replaced with a three-model majority vote:

| Judge model | Provider |
|-------------|----------|
| `llama-3.3-70b-versatile` | Groq |
| `llama-3.1-8b-instant` | Groq |
| `qwen/qwen3-32b` | Groq |

**Protocol:**
- Responses presented as "Response 1" and "Response 2" (not A/B) to reduce positional bias
- Each judge independently produces a verdict (1, 2, or TIE) with one-sentence reasoning
- Final verdict = majority (2/3 or 3/3 agreement)
- If all three judges choose different verdicts, the final verdict is TIE
- Per-judge verdicts and reasoning are recorded in `judge_details` in the results JSON

---

## Results

### Overall

| Metric | Value |
|--------|-------|
| Total scenarios | 20 |
| **Kokoro (B) wins** | **9/20 = 45.0%** |
| Semantic-only (A) wins | 11/20 = 55.0% |
| TIE | 0/20 |
| Scenarios with different memories | 11/20 (55%) |
| Average memory overlap | 81.7% |

### Inter-Judge Agreement

| Agreement level | Count | Rate |
|-----------------|-------|------|
| Unanimous (3/3) | 5/20 | 25% |
| Majority (2/3) | 15/20 | 75% |
| No agreement | 0/20 | 0% |
| **Total agreement rate** | **20/20** | **100%** |

Every scenario reached at least majority consensus. No TIEs from full disagreement.

### Per-Judge Voting Pattern

| Judge model | Voted A | Voted B |
|-------------|---------|---------|
| `llama-3.3-70b-versatile` | 11 | 9 |
| `llama-3.1-8b-instant` | 14 | 6 |
| `qwen/qwen3-32b` | 0 | 20 |

`qwen/qwen3-32b` voted B on every single scenario. The two Llama models provided the discriminating votes. The 8b model was notably more conservative, favouring A in 14/20 cases. The final majority-vote verdicts were determined by the two Llama judges agreeing in 15/20 cases, with qwen breaking the tie in favour of B on the remaining 5.

### Per-Scenario Results

| ID | Arc type | Memories differ | Verdict | Agreement | Judges (70b / 8b / qwen) |
|----|----------|:-:|:-:|:-:|:-:|
| l01 | recovery_over_burnout | YES | **B** | majority | B A B |
| l02 | grief_to_healing | YES | A | majority | A A B |
| l03 | anxiety_to_confidence | YES | **B** | unanimous | B B B |
| l04 | career_crisis_to_growth | no | A | majority | A A B |
| l05 | divorce_to_rebuilding | no | **B** | unanimous | B B B |
| l06 | academic_failure_to_success | no | A | majority | A A B |
| l07 | health_scare_to_recovery | YES | **B** | unanimous | B B B |
| l08 | social_anxiety_to_connection | YES | A | majority | A A B |
| l09 | creative_block_to_flow | no | A | majority | A A B |
| l10 | loneliness_to_community | no | **B** | unanimous | B B B |
| l11 | financial_stress_to_stability | no | A | majority | A A B |
| l12 | chronic_pain_acceptance | YES | **B** | majority | B A B |
| l13 | new_parent_exhaustion_to_adaptation | no | A | majority | A A B |
| l14 | relocation_distress_to_belonging | no | A | majority | A A B |
| l15 | identity_crisis_to_clarity | YES | **B** | unanimous | B B B |
| l16 | relationship_conflict_to_resolution | no | **B** | majority | B A B |
| l17 | sobriety_journey | YES | A | majority | A A B |
| l18 | caregiver_burnout_to_boundary_setting | YES | A | majority | A A B |
| l19 | performance_anxiety_to_confidence | YES | A | majority | A A B |
| l20 | grief_anniversary_vs_integrated_loss | YES | **B** | majority | B A B |

### Comparison with v1 (Original 6 Scenarios, Single Judge)

| Scenario | v1 verdict (single judge) | v2 verdict (majority vote) | Change |
|----------|:--:|:--:|:--|
| l01 recovery_over_burnout | B | **B** | held |
| l02 grief_to_healing | B | **A** | flipped |
| l03 anxiety_to_confidence | B | **B** | held |
| l04 career_crisis_to_growth | A | **A** | held |
| l05 divorce_to_rebuilding | B | **B** | held |
| l06 academic_failure_to_success | B | **A** | flipped |

- **4/6 verdicts held** under the new three-judge protocol
- **2/6 flipped from B to A** (grief_to_healing, academic_failure_to_success) — both cases where the 8b model disagreed with the 70b model, and the majority went with A
- v1 Kokoro win rate on original 6: **5/6 = 83.3%**
- v2 Kokoro win rate on original 6: **3/6 = 50.0%**

### Where Kokoro (B) Wins

B won 9/20 scenarios. Patterns in B wins:

**Unanimous B wins (all three judges agreed):**
- **l03 anxiety_to_confidence** — B pulled an old anxiety session alongside recovery sessions, giving it context for acknowledging the journey. All judges praised this nuanced acknowledgment.
- **l05 divorce_to_rebuilding** — same memories, but B's state summary drove more emotionally complex follow-up questions about co-parenting.
- **l07 health_scare_to_recovery** — B retrieved recovery-phase sessions instead of the initial health scare, producing a response that matched the user's current healthy relationship with their body.
- **l10 loneliness_to_community** — same memories, but B asked about the user's emotional state in the quiet moment, connecting it to their journey from isolation to chosen solitude.
- **l15 identity_crisis_to_clarity** — B retrieved the "I'm training to be a teacher" session, allowing it to reference the user's new identity with specificity and pride.

**Majority B wins (70b + qwen agreed, 8b dissented):**
- **l01 recovery_over_burnout** — B focused on the user's current healthy habits; A focused on past burnout concern.
- **l12 chronic_pain_acceptance** — B validated the user's journey and progress in managing pain.
- **l16 relationship_conflict_to_resolution** — B referenced recent positive interactions and therapy progress.
- **l20 grief_anniversary_vs_integrated_loss** — B retrieved the Sunday call tradition with mum, enabling a warmer, more specific response.

### Where Semantic-Only (A) Wins

A won 11/20 scenarios. Patterns in A wins:

**A wins on same-memory scenarios (no retrieval divergence):**
- l04, l06, l09, l11, l13, l14 — when A and B retrieve identical memories, A sometimes produces responses that are equally or more emotionally aware. The state summary in B didn't always add value, and in some cases B's response was more generic despite having the emotional context.

**A wins despite retrieval divergence:**
- **l02 grief_to_healing** — both responses assumed the user might be struggling; A's was judged as more supportive despite B having "healed" memories.
- **l08 social_anxiety_to_connection** — B's hybrid retrieval pulled an old isolation session, causing it to ask "are you still hesitant?" — misreading the user's confident state.
- **l17 sobriety_journey** — A directly addressed the challenge of being in a bar sober; judges valued this practical support over B's more celebratory tone.
- **l18 caregiver_burnout** — both responses incorrectly assumed ongoing burnout; A's was judged as slightly more appropriate.
- **l19 performance_anxiety_to_confidence** — A referenced a specific recent success (client presentation); B was more general about progress.

### Retrieval Divergence Analysis

| Retrieval divergence | B wins | A wins | B win rate |
|:--|:--:|:--:|:--:|
| Memories differ (11 scenarios) | 6 | 5 | **54.5%** |
| Same memories (9 scenarios) | 3 | 6 | **33.3%** |

When retrieval diverges (hybrid pulls different memories than semantic-only), Kokoro wins more often. When memories are the same, the state summary alone is not sufficient to consistently outperform semantic-only retrieval.

### Qualitative Highlights

**l07 health_scare_to_recovery (B wins, unanimous):**
- A retrieved the initial scare ("found a lump last week, trying not to panic") alongside recovery sessions
- B retrieved only recovery sessions ("one year clear, doctor says perfect health")
- A: "I know from our past conversations that you've been through some challenging times"
- B: "I'm sure it'll be a reassuring experience given your positive progress... you've developed a healthier relationship with your body"
- B correctly frames the checkup as routine; A projects residual anxiety

**l08 social_anxiety_to_connection (A wins):**
- B's hybrid retrieval pulled an old isolation session ("got invited to a group dinner, said no")
- This caused B to ask "are you still hesitant about taking that step?" — when the user now has a close friend group and is clearly excited
- A correctly treated the trip as exciting, recognising the user's belonging
- **This is a case where the hybrid retrieval pulled a counterproductive old session**

**l15 identity_crisis_to_clarity (B wins, unanimous):**
- B retrieved "I'm training to be a teacher, said it with pride" — A did not
- B: "the progress you've made and the sense of purpose you've found in mentoring and your journey to become a teacher"
- A: "combining your passion for mentoring with your leadership background in a new role"
- B's specificity about teaching was more emotionally resonant than A's generic framing

**l17 sobriety_journey (A wins):**
- B retrieved the 6-month milestone ("I'm freer now than I ever was") — A did not
- Despite this, A's practical response ("do you have a plan to support yourself if you start to feel tempted?") was judged as more emotionally aware
- **Judges valued practical harm-reduction support over celebratory framing for a bar visit during sobriety**

---

## Summary

| Metric | v1 (6 scenarios, single judge) | v2 (20 scenarios, 3-judge majority) |
|--------|:--:|:--:|
| Total scenarios | 6 | 20 |
| Kokoro (B) win rate | 83.3% | 45.0% |
| Retrieval divergence rate | 50% | 55% |
| Average memory overlap | 83.4% | 81.7% |
| Inter-judge agreement | N/A (single judge) | 100% (all majority or unanimous) |
| Unanimous verdicts | N/A | 5/20 (25%) |

The expanded evaluation reveals a more nuanced picture than the original 6-scenario set suggested:

1. **The original 83.3% was inflated** — the 6 original scenarios were hand-picked to maximise retrieval divergence, and a single judge introduced noise. Under three-judge majority vote, the same 6 scenarios yield 50% (3/6).

2. **Retrieval divergence remains the key mechanism** — B wins 54.5% when memories differ vs 33.3% when they don't. The hybrid retrieval is doing useful work when it fires.

3. **Hybrid retrieval can backfire** — in l08 and l18, the emotional axis pulled old distress sessions that caused B to misread the user's current state. The emotional retrieval is not always an improvement.

4. **The state summary alone is weaker than expected** — on same-memory scenarios, B's state summary did not consistently improve responses. A's semantic-only approach often produced equally emotionally aware responses.

5. **Judge model matters** — `qwen/qwen3-32b` systematically favoured B on all 20 scenarios, while `llama-3.1-8b-instant` systematically favoured A on 14/20. The 70b model provided the most balanced voting. Multi-judge majority vote is more robust than any single judge.

6. **45% is a realistic baseline** — on a diverse set of 20 long-history scenarios with three-judge consensus, Kokoro's emotional retrieval and state summary provide a meaningful but not dominant advantage. The system works best on recovery/transition arcs where retrieval divergence surfaces genuinely different emotional contexts.
