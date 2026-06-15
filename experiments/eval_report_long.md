# Kokoro Long-History Retrieval Evaluation — Full Results

**Experiment:** Long-history scenarios (10–12 sessions) designed to surface
retrieval divergence between semantic-only and hybrid retrieval.  
**Emotional axis:** VAD-coordinate L2 distance (more discriminative than state cosine)  
**Model:** `llama-3.3-70b-versatile` (Groq)  
**Date:** 2026-06-13  

---

## Summary

| Metric | Value |
|--------|-------|
| Kokoro (B) win rate | **5/6 = 83.3%** |
| Semantic (A) wins | 1/6 |
| Scenarios with different memories (A vs B) | 3/6 (50%) |
| Average memory overlap | 83.4% |
| Sessions per scenario | 10–12 |
| Standard run (20 scenarios) | 55.0% win rate, 5% memory divergence |

**Key finding:** With longer histories (10-12 sessions) and VAD-based
emotional scoring, retrieval divergence rises to 50% of scenarios
and Kokoro's win rate rises to **83.3%** vs 55% on the standard 20-scenario set.

### Arc-level results

| Arc type | Memories differ | Verdict |
|----------|-----------------|---------|
| 🔥→🌿 `recovery_over_burnout` | YES | B (Kokoro) |
| 🕯️→💛 `grief_to_healing` | YES | B (Kokoro) |
| 😰→💪 `anxiety_to_confidence` | YES | B (Kokoro) |
| 🧱→🚀 `career_crisis_to_growth` | no | A (semantic) |
| 💔→🏡 `divorce_to_rebuilding` | no | B (Kokoro) |
| 📉→🎓 `academic_failure_to_success` | no | B (Kokoro) |

---

## Scenario-by-scenario results

### l01 🔥→🌿 `recovery_over_burnout`

**User:** Marcus burned out as a senior dev, went through therapy and boundary-setting, and now genuinely enjoys his work again  
**New message:** *"big sprint kicking off next week"*  
**Expected awareness:** user is in a genuinely recovered state — this sprint should feel manageable and even exciting, not threatening. Good companion should respond with energy and forward-looking curiosity, not cautionary concern about burnout  
**State:** valence=+0.246  arousal=+0.360  trend=improving  
**Memory overlap:** 67%  ⚠️ **Retrieval differs — B pulls recent recovery sessions, A pulls old distress sessions**  
**Verdict:** 🟢 Kokoro wins  

**State summary (injected into B only):**
> User shows signs of emotional improvement across the past 12 sessions, with valence notably rising toward a positive range (current activated activation). Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Retrieved memories — Condition A (semantic-only):**
1. _work has been a lot lately. sprint got moved up and i've got two features due same week i'm fine. just need to push through. alway…_
2. _got back into the gym. three times this week. sleep is good. feeling solid it really is. therapy helped me see the patterns. i kno…_
3. _finished the sprint but felt nothing. just empty. not relieved, just hollow never like this. i used to get that buzz after shippin…_

**Retrieved memories — Condition B (hybrid, VAD emotional axis):**
1. _work has been a lot lately. sprint got moved up and i've got two features due same week i'm fine. just need to push through. alway…_
2. _got back into the gym. three times this week. sleep is good. feeling solid it really is. therapy helped me see the patterns. i kno…_
3. _team shipped a big feature this week. i feel proud of it. genuinely, not just relieved it's done a much better one. this job can b…_  ← **emotionally matched (not in A)**

**Response A (semantic only):**
> I'm a bit concerned about you taking on another big sprint so soon, especially after feeling empty and hollow after the last one. Are you feeling prepared and energized for this next challenge, or are you still recovering from the previous sprint?

**Response B (Kokoro — state summary + hybrid memories):**
> You've been preparing well, both physically and mentally, and that's great to see. Remembering your warning signs and maintaining a healthy balance will be crucial as you head into this big sprint. How are you feeling about taking on this new challenge, and is there anything specific that's on your mind?

**Judge reasoning:**
> Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the user's past struggles with burnout and recovery, and offers a supportive and empowering message that takes into account their growth and progress.

---

### l02 🕯️→💛 `grief_to_healing`

**User:** Priya went through a devastating breakup, grieved deeply for months, rebuilt herself, and is now in a healthy new relationship  
**New message:** *"heard his favourite song today"*  
**Expected awareness:** user is genuinely healed — hearing the ex's song is a small nostalgic moment, not a crisis. Good companion should meet her where she actually is (stable, self-aware, happy) rather than treating it as a grief relapse trigger  
**State:** valence=+0.178  arousal=+0.125  trend=improving  
**Memory overlap:** 67%  ⚠️ **Retrieval differs — B pulls recent recovery sessions, A pulls old distress sessions**  
**Verdict:** 🟢 Kokoro wins  

**State summary (injected into B only):**
> User shows signs of emotional improvement across the past 11 sessions, with valence notably rising toward a positive range (current moderate activation). Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Retrieved memories — Condition A (semantic-only):**
1. _couldn't go to work today. just cried all morning. this is pathetic i keep finding his stuff around the flat. a hoodie, a mug. eve…_
2. _met someone. nothing serious but we've been on three dates and i genuinely like him he's curious. asks questions about me. listens…_
3. _went to our favourite restaurant with a friend. bad idea. had to leave i hate that he's taken all these places from me. that was m…_

**Retrieved memories — Condition B (hybrid, VAD emotional axis):**
1. _met someone. nothing serious but we've been on three dates and i genuinely like him he's curious. asks questions about me. listens…_
2. _i think i'm happy. actually happy. not pretending to be fine, just actually good strange and wonderful. i'm grateful for what i we…_  ← **emotionally matched (not in A)**
3. _couldn't go to work today. just cried all morning. this is pathetic i keep finding his stuff around the flat. a hoodie, a mug. eve…_

**Response A (semantic only):**
> That must have been really tough for you - it's like the memories are still lingering everywhere. It's okay to feel overwhelmed, and it's great that you're acknowledging your emotions. Remember, it's a process, and you're doing the best you can, and that's something to be proud of.

**Response B (Kokoro — state summary + hybrid memories):**
> That can be a powerful trigger for emotions, especially when it's connected to someone special. How did you feel when you heard his favorite song, and did it bring back any happy memories from your time together?

**Judge reasoning:**
> Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the potential emotional trigger and invites the user to reflect on their feelings, while also considering the complexity of their emotional history and the possibility of both positive and negative memories.

---

### l03 😰→💪 `anxiety_to_confidence`

**User:** Kaveh struggled with severe social anxiety, avoided parties and crowded spaces, went through CBT therapy, and now genuinely enjoys social events  
**New message:** *"big work party on Friday, apparently everyone's going"*  
**Expected awareness:** user now enjoys social events and is socially confident — this should feel like a fun Friday plan, not a threat. Good companion should be interested and upbeat, not cautious about anxiety triggers  
**State:** valence=+0.039  arousal=+0.169  trend=improving  
**Memory overlap:** 67%  ⚠️ **Retrieval differs — B pulls recent recovery sessions, A pulls old distress sessions**  
**Verdict:** 🟢 Kokoro wins  

**State summary (injected into B only):**
> User shows signs of emotional improvement across the past 11 sessions, with valence gradually rising toward a positive range (current moderate activation). Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Retrieved memories — Condition A (semantic-only):**
1. _hosted a dinner party for eight people. my flat. my cooking. had a great evening it really is. i thought about that halfway throug…_
2. _got invited to another party this weekend and my reaction was just 'cool, what time'. didn't dread it at all i know. it almost snu…_
3. _went to a party. knew almost nobody. stayed two hours. enjoyed myself honestly? the conversations. i'd forgotten i can be interest…_

**Retrieved memories — Condition B (hybrid, VAD emotional axis):**
1. _hosted a dinner party for eight people. my flat. my cooking. had a great evening it really is. i thought about that halfway throug…_
2. _got invited to another party this weekend and my reaction was just 'cool, what time'. didn't dread it at all i know. it almost snu…_
3. _got invited to a party this weekend and i've been dreading it since Tuesday don't know what to say. standing around with a drink l…_  ← **emotionally matched (not in A)**

**Response A (semantic only):**
> It sounds like you're feeling pretty calm about it, which is great - a big change from how you might have reacted in the past. I'm curious, what do you think has helped you become more comfortable in social situations like this? Are you looking forward to it, or just taking it in stride?

**Response B (Kokoro — state summary + hybrid memories):**
> It sounds like you're feeling a bit apprehensive about the party, but I've noticed how far you've come in recent sessions - you've been handling social situations with more ease and confidence. Remember how you enjoyed hosting your dinner party and even looked forward to another party recently? You've made significant progress, and I'm here to support you through any nerves you might be feeling now.

**Judge reasoning:**
> Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the user's past struggles with social anxiety, recognizes their progress, and offers support in a way that is tailored to their specific history and experiences.

---

### l04 🧱→🚀 `career_crisis_to_growth`

**User:** Diane hated her corporate job for years, was terrified to leave, finally made the leap to a startup, and is now thriving  
**New message:** *"performance review is next week"*  
**Expected awareness:** user is thriving and confident in her new role — review should feel exciting, a chance to showcase what she's built. Good companion should be enthusiastic and forward-looking, not project anxiety from the old job context  
**State:** valence=-0.062  arousal=+0.067  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (injected into B only):**
> User shows signs of emotional improvement across the past 11 sessions, with valence gradually rising toward a neutral range (current moderate activation).

**Retrieved memories — Condition A (semantic-only):**
1. _first week at the new place. it's chaotic and fast and I love it my opinion actually matters here. I sat in a meeting and they lis…_
2. _had an interview. first one in seven years. I was so nervous I could barely speak in the first five minutes I settled down eventua…_
3. _Monday mornings are different now. I actually look forward to them. That's new I know who I am at work now. That was the thing I'd…_

**Retrieved memories — Condition B (hybrid, VAD emotional axis):**
1. _Monday mornings are different now. I actually look forward to them. That's new I know who I am at work now. That was the thing I'd…_
2. _first week at the new place. it's chaotic and fast and I love it my opinion actually matters here. I sat in a meeting and they lis…_
3. _had an interview. first one in seven years. I was so nervous I could barely speak in the first five minutes I settled down eventua…_

**Response A (semantic only):**
> You must be feeling a mix of excitement and nerves about your performance review next week. It's great that you've found your footing and feel heard in your new role, and I'm sure that confidence will shine through in your review. What are your hopes and expectations for the review - are there any specific areas you're looking to discuss or goals you want to set?

**Response B (Kokoro — state summary + hybrid memories):**
> I'm so proud of the progress you've made, and I think this performance review is a great opportunity to acknowledge your growth and the value you've been bringing to your new role. Remember how far you've come, from feeling lost to now being heard and valued in your workplace - that's something to celebrate. What are your hopes and expectations for this review?

**Judge reasoning:**
> Response A shows more emotional awareness and understanding of the user's situation because it acknowledges the user's potential nervousness about the performance review, in addition to their growth and confidence, demonstrating a more nuanced understanding of their emotional state.

---

### l05 💔→🏡 `divorce_to_rebuilding`

**User:** Tom went through a brutal divorce, struggled with isolation and co-parenting stress, and has now rebuilt a stable, fulfilling single-parent life  
**New message:** *"taking the kids this weekend, got a full plan"*  
**Expected awareness:** user is a stable, confident single parent who enjoys his time with the kids — this should be met with warmth and enthusiasm for the plans, not concern about co-parenting stress or divorce grief  
**State:** valence=-0.043  arousal=+0.095  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (injected into B only):**
> User shows signs of emotional improvement across the past 11 sessions, with valence gradually rising toward a neutral range (current moderate activation).

**Retrieved memories — Condition A (semantic-only):**
1. _have the kids every other week now and I love my weeks with them and I actually use my weeks without them went to the theatre last…_
2. _divorce is finalised. fifteen years. I didn't know it was possible to feel this empty the house. coming back to it without them in…_
3. _she and I used a mediator to sort out the co-parenting schedule properly. it's actually working no more arguments at pickup. we ke…_

**Retrieved memories — Condition B (hybrid, VAD emotional axis):**
1. _she and I used a mediator to sort out the co-parenting schedule properly. it's actually working no more arguments at pickup. we ke…_
2. _have the kids every other week now and I love my weeks with them and I actually use my weeks without them went to the theatre last…_
3. _divorce is finalised. fifteen years. I didn't know it was possible to feel this empty the house. coming back to it without them in…_

**Response A (semantic only):**
> That sounds wonderful, I'm so glad to hear you have a fun-filled weekend planned with the kids. I'm sure they're looking forward to spending time with you, and it's great that you get to create new memories together. What are some of the exciting things you have planned for the weekend?

**Response B (Kokoro — state summary + hybrid memories):**
> It's wonderful that you have a fun-filled weekend planned with the kids, and having something to look forward to can be really uplifting. I'm sure they'll love spending time with you, and it's great that you're making the most of your time together. How are you feeling about having them this weekend, are you excited to see them after their week with their mom?

**Judge reasoning:**
> Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the potential emotional complexity of co-parenting and invites the user to express their feelings, whereas Response A is more focused on the activities planned.

---

### l06 📉→🎓 `academic_failure_to_success`

**User:** Amara was failing her second year of university, seriously considered dropping out, rebuilt her study approach with support, and is now excelling  
**New message:** *"dissertation results tomorrow"*  
**Expected awareness:** user has genuinely excelled and her supervisor said her work is outstanding — results should feel like confirmation of earned success, not a source of anxiety. Good companion should match her earned confidence, not treat this as a stress point  
**State:** valence=+0.054  arousal=-0.091  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (injected into B only):**
> User shows signs of emotional improvement across the past 11 sessions, with valence notably rising toward a positive range (current low-energy activation). Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Retrieved memories — Condition A (semantic-only):**
1. _submitted my dissertation chapter draft. ahead of schedule. felt good writing it I enjoy it now. the research, the writing. I reme…_
2. _supervisor said my dissertation is one of the strongest she's supervised this year I cried a bit. in a good way. I almost left and…_
3. _studied for two hours today. properly. understood what I was reading. first time in months Pomodoro timers, no phone, one chapter …_

**Retrieved memories (Condition B):** _(same as A)_

**Response A (semantic only):**
> I'm sending you all my positive thoughts for tomorrow's results - you've worked so hard and deserve to feel proud of yourself, no matter the outcome. Remember to be kind to yourself and take a moment to breathe, as it's a big day ahead. You've got this, and I'm here cheering you on.

**Response B (Kokoro — state summary + hybrid memories):**
> You must be feeling a mix of excitement and nerves about getting your dissertation results tomorrow. It's great that you've been making progress and feeling more confident about your work, and I'm hoping the results will be a positive reflection of your efforts. How are you planning to take care of yourself tomorrow, regardless of the outcome?

**Judge reasoning:**
> Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the user's potential emotional state and references their progress, while also inquiring about their self-care, demonstrating a deeper understanding of their emotional history and needs.

---
