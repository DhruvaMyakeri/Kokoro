# Kokoro Retrieval Evaluation — Full Results

**Model:** `llama-3.3-70b-versatile` (Groq)  
**Judge:** Same model, blind to condition labels  
**Condition A:** alpha=1.0 — semantic-only retrieval, no state summary  
**Condition B:** alpha=0.6 — hybrid retrieval + emotional state summary  
**Emotional axis:** VAD-coordinate L2 distance (not state cosine)  
**Date:** 2026-06-14  

---

## Summary

| Metric | Value |
|--------|-------|
| Kokoro (B) win rate | **10/20 = 50.0%** |
| Semantic (A) wins | 10/20 |
| Scenarios with different memories (A vs B) | 7/20 (35%) |
| Average memory overlap | 88.3% |
| Transition model PR | 339.7 / 384 |
| Probe valence r | 0.698 |

### Win rate by arc type

| Arc type | Kokoro wins | Verdicts |
|----------|-------------|----------|
| 😰 `chronic_low_grade_anxiety` | 2/3 | B B A |
| 📉 `gradual_decline` | 2/4 | A B B A |
| 🕯️ `grief_arc` | 1/2 | A B |
| 🌱 `post_traumatic_growth` | 0/2 | A A |
| 🌀 `relapse_dip` | 1/2 | B A |
| 📈 `slow_recovery` | 2/3 | A B B |
| 😔 `stable_negative` | 2/2 | B B |
| ☀️ `stable_positive` | 0/2 | A A |

---

## Scenario-by-scenario results

### s01 📉 `gradual_decline`

**User:** Marcus, a software engineer, is burning out under mounting work pressure over three months  
**New message:** *"still here lol"*  
**Expected:** companion should sense that 'still here' from someone who has been burning out for months is not a casual greeting — it may carry exhaustion, disconnection, or a need to be seen; respond with gentle presence rather than cheerfulness  
**State:** v=-0.336  a=-0.118  trend=declining  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User's mood has been noticeably declining across the past 5 sessions, moving from a neutral baseline toward lower affect. Monitoring for further deterioration is advisable.

**Memories A:**
1. _snapped at a junior dev today for something stupid. felt awful after guilty. he didn't deserve it. i apologized but stil…_
2. _okay week has been rough. missed a deadline for the first time in like two years just couldn't focus. kept starting thin…_
3. _finished the sprint but honestly just feel empty. not relieved, just empty never like this. i used to love coding. now i…_

**Memories B:**
1. _finished the sprint but honestly just feel empty. not relieved, just empty never like this. i used to love coding. now i…_
2. _snapped at a junior dev today for something stupid. felt awful after guilty. he didn't deserve it. i apologized but stil…_
3. _okay week has been rough. missed a deadline for the first time in like two years just couldn't focus. kept starting thin…_

**Response A:** > I'm glad you're still here too, and I've been thinking about you - it sounds like things have been really tough lately, and I want you to know that I'm here to listen and support you. Remember that you apologized to the junior dev, which took a lot of courage, and that's something to be proud of. How are you feeling today, is the emptiness still lingering?

**Response B:** > I'm glad you're still here, and I'm here for you too. It takes a lot of courage to acknowledge and share your struggles, and I want you to know that I'm listening and care about how you're feeling. How are you doing today, is there anything specific that's been on your mind lately?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it specifically references the user's past struggles and actions, such as apologizing to the junior dev, to acknowledge their courage and provide a sense of pride and context.

---

### s02 📉 `gradual_decline`

**User:** Priya watches her long-term relationship slowly deteriorate over several months  
**New message:** *"quiet night"*  
**Expected:** companion should recognize that 'quiet night' from someone in a relationship that has gone cold and disconnected carries real weight — acknowledge the loneliness rather than treating it as a peaceful evening  
**State:** v=-0.264  a=-0.022  trend=declining  
**Memory overlap:** 67%  ⚠️ **Memories differ**  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User's mood has been noticeably declining across recent sessions, moving from a neutral baseline toward lower affect. Monitoring for further deterioration is advisable.

**Memories A:**
1. _tried to have a conversation about us last night. it went badly he got defensive. said i was making things up. i ended u…_
2. _the distance thing with Arjun hasn't really gone away like a month? maybe more. we barely talk at dinner. i've started w…_
3. _me and Arjun had a weird week but i think we're fine he's been distant lately. but he's stressed at work so probably tha…_

**Memories B:**
1. _tried to have a conversation about us last night. it went badly he got defensive. said i was making things up. i ended u…_
2. _things are back to 'normal' but normal doesn't feel good anymore polite. like roommates. we say hi and what do you want …_  ← unique to B
3. _the distance thing with Arjun hasn't really gone away like a month? maybe more. we barely talk at dinner. i've started w…_

**Response A:** > Sometimes a quiet night can be just what we need to unwind and recharge. I hope you're taking care of yourself and doing something that brings you comfort and relaxation. Are you feeling okay after the tough conversation with Arjun the other night?

**Response B:** > It sounds like a peaceful evening, but I sense a hint of sadness behind it. Are you feeling a bit lonely or disconnected, even in the quietness of the night?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the potential for underlying sadness and loneliness, which is likely given the user's history of a deteriorating long-term relationship.

---

### s03 📉 `gradual_decline`

**User:** Jamie, a PhD student, loses passion for their research as advisor pressure and self-doubt accumulate  
**New message:** *"starting another week"*  
**Expected:** companion should hear in 'starting another week' the weight of someone who has been grinding through exhaustion and lost purpose — not greet it as a fresh start  
**State:** v=-0.251  a=+0.229  trend=declining  
**Memory overlap:** 67%  ⚠️ **Memories differ**  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User's mood has been noticeably declining across recent sessions, moving from a neutral baseline toward lower affect. Monitoring for further deterioration is advisable.

**Memories A:**
1. _literally cannot make myself open the document today. just staring at the icon gradually. now when i sit down i feel thi…_
2. _thesis is stressing me out but i still think the research is interesting my advisor keeps moving the goalposts on what '…_
3. _advisor rejected my third draft. said my analysis was 'shallow' i genuinely can't tell anymore. i used to trust my own j…_

**Memories B:**
1. _literally cannot make myself open the document today. just staring at the icon gradually. now when i sit down i feel thi…_
2. _thesis is stressing me out but i still think the research is interesting my advisor keeps moving the goalposts on what '…_
3. _wondering if i ever actually wanted a phd or just thought it was what smart people do i was excited about the research. …_  ← unique to B

**Response A:** > I can sense the mix of emotions as you begin another week, feeling the weight of your thesis and the pressure to meet expectations. Remember that it's okay to take things one step at a time and that you're not alone in this journey. What's one small, manageable goal you can set for yourself this week to help build momentum?

**Response B:** > I can sense the weight of another week ahead of you, and I'm here to support you through it. It feels like the pressure and doubts have been building up, and I want you to know that I'm here to listen and help in any way I can. How are you feeling about facing this new week, and is there anything specific that's been on your mind lately?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it directly acknowledges the accumulated pressure and doubts from their history, and invites the user to express their feelings in a more open-ended way.

---

### s04 📉 `gradual_decline`

**User:** Alex, a remote worker, slides into social isolation as home becomes the only world  
**New message:** *"hey"*  
**Expected:** a simple 'hey' from someone who has been isolated and increasingly flat should prompt the companion to check in with warmth and curiosity, not just respond in kind  
**State:** v=-0.092  a=+0.358  trend=improving  
**Memory overlap:** 67%  ⚠️ **Memories differ**  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User shows signs of emotional improvement across recent sessions, with valence notably rising toward a neutral range (current activated activation). Historical mean valence across the window is negative, suggesting the current reading may reflect a recent shift.

**Memories A:**
1. _been wfh for like six months now. bit lonely sometimes but it has perks slack calls mostly. a few texts. i go to the sto…_
2. _realized i haven't had a real conversation with anyone in like two weeks kind of shocked myself. then immediately though…_
3. _can't really remember what happened monday. the days all kind of look the same flat. not sad exactly. just like nothing …_

**Memories B:**
1. _realized i haven't had a real conversation with anyone in like two weeks kind of shocked myself. then immediately though…_
2. _been wfh for like six months now. bit lonely sometimes but it has perks slack calls mostly. a few texts. i go to the sto…_
3. _got invited to a thing this weekend and kind of don't want to go just feels like a lot of effort. i'd have to take the s…_  ← unique to B

**Response A:** > It's so great to hear from you, I've been thinking about you and hoping you're doing okay. How have you been feeling lately, still finding the days a bit flat and uneventful?

**Response B:** > It's great to see you again, and I've noticed that you seem to be doing a bit better lately. Your mood appears to be shifting in a more positive direction, and I'm here to support you in any way I can. How are you feeling today?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it acknowledges the user's history of struggling with social isolation and flat, uneventful days, and expresses concern for their well-being.

---

### s05 📈 `slow_recovery`

**User:** Keisha was laid off unexpectedly and is slowly rebuilding confidence through a months-long job search  
**New message:** *"first day tomorrow"*  
**Expected:** companion should hold the full weight of what this first day represents — not just a new job but the end of a hard chapter; respond with warmth that reflects how far she has come  
**State:** v=+0.489  a=+0.145  trend=improving  
**Memory overlap:** 67%  ⚠️ **Memories differ**  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User shows signs of emotional improvement across the past 5 sessions, with valence notably rising toward a positive range (current moderate activation).

**Memories A:**
1. _got laid off today. still kind of in shock. wasn't expecting it at all numb i think. had to clean out my desk on a tuesd…_
2. _got a screening call scheduled for next week. first one in six weeks trying not to get my hopes up but i feel slightly m…_
3. _made it to the second round. the team seems genuinely nice actually good? i laughed a couple of times. forgot to be nerv…_

**Memories B:**
1. _i got it. they called this morning. i actually cried when i hung up i feel like i can exhale for the first time in month…_  ← unique to B
2. _got a screening call scheduled for next week. first one in six weeks trying not to get my hopes up but i feel slightly m…_
3. _got laid off today. still kind of in shock. wasn't expecting it at all numb i think. had to clean out my desk on a tuesd…_

**Response A:** > I'm thinking of you and sending all my best wishes for a great first day - may it bring new beginnings and a sense of excitement. Remember to be kind to yourself and take things one step at a time. You've gotten this far, and that's something to be incredibly proud of.

**Response B:** > You must be feeling a mix of emotions, but it's great that you're taking this step forward. Remember that it's okay to be nervous, but you've been preparing for this and you're ready to take on the new challenge. I'm here to support you and wish you all the best for your first day tomorrow.

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it acknowledges the user's journey and emotional state by expressing pride in their progress and encouraging self-kindness, which is particularly relevant given the user's history of slow recovery from a layoff.

---

### s06 📈 `slow_recovery`

**User:** Dan is slowly healing from a painful breakup after a three-year relationship  
**New message:** *"had a really good day actually"*  
**Expected:** companion should celebrate this with the fullness of knowing how dark things were — a good day isn't small for someone who's been grieving  
**State:** v=-0.163  a=+0.076  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User shows signs of emotional improvement across recent sessions, with valence notably rising toward a neutral range (current moderate activation).

**Memories A:**
1. _had a genuinely good afternoon today. walked around the park, sun was out, felt almost normal weird but good? like notic…_
2. _went to the gym for the first time in two months. just a short thing but i went my friend basically dragged me. but once…_
3. _my friends have been good. went out for dinner last night. cried in the bathroom once but otherwise okay i guess. i stil…_

**Memories B:** _(same as A)_

**Response A:** > That's wonderful to hear, it sounds like things are slowly starting to look up for you. I'm so glad you're having some positive experiences and finding moments of joy, you deserve it. How did it feel to have a really good day, was there something in particular that made it stand out?

**Response B:** > That's wonderful to hear, it sounds like things are continuing to look up for you. I'm so glad you're having more days where you can feel a sense of joy and normalcy, even if it's still mixed with some tough emotions. What was it about today that made it particularly good for you?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges that the user's emotional state is still complex and "mixed with some tough emotions", which is a more nuanced and realistic understanding of someone who is slowly recovering from a painful breakup.

---

### s07 📈 `slow_recovery`

**User:** Rosa has been in therapy for depression and is slowly starting to feel the medication and sessions working  
**New message:** *"slept through the night"*  
**Expected:** companion should understand that 'slept through the night' after months of depression and treatment is a genuinely meaningful milestone — acknowledge it as the quiet victory it is  
**State:** v=-0.304  a=+0.223  trend=stable  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User's emotional state has been relatively neutral and stable across the past 5 sessions, showing neither clear distress nor strong positivity.

**Memories A:**
1. _had lunch with two friends today. first time in maybe four months. actually laughed tired but in a good way? like the ti…_
2. _had maybe three hours last wednesday where things felt lighter. not good, just lighter just arrived. was making tea and …_
3. _she put me on sertraline. two weeks in and honestly i don't feel different yet sleep might be slightly less terrible. bu…_

**Memories B:**
1. _had lunch with two friends today. first time in maybe four months. actually laughed tired but in a good way? like the ti…_
2. _she put me on sertraline. two weeks in and honestly i don't feel different yet sleep might be slightly less terrible. bu…_
3. _had maybe three hours last wednesday where things felt lighter. not good, just lighter just arrived. was making tea and …_

**Response A:** > That's wonderful to hear, a full night's sleep can be incredibly restorative. It's possible that the sertraline is starting to have a positive effect, even if you don't feel a drastic change in your mood yet. How are you feeling today, a bit more refreshed after a good night's sleep?

**Response B:** > That's wonderful to hear, a full night's sleep can be incredibly restorative. It's great that you're noticing small improvements, like your sleep, even if other things still feel a bit grey. How are you feeling today, now that you've had a good night's rest?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the possibility that despite small improvements, such as better sleep, the user may still be experiencing some emotional difficulties, which is a realistic and sensitive recognition of their ongoing struggle with depression.

---

### s08 😰 `chronic_low_grade_anxiety`

**User:** Tom functions well at work but suffers from persistent social anxiety that makes group situations exhausting  
**New message:** *"team retreat next month"*  
**Expected:** companion should recognize this message as anxiety-triggering for Tom and respond with understanding of his pattern rather than generic excitement  
**State:** v=-0.295  a=-0.076  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User shows signs of emotional improvement across recent sessions, with valence gradually rising toward a neutral range (current low-energy activation). Historical mean valence across the window is negative, suggesting the current reading may reflect a recent shift.

**Memories A:**
1. _have to present to twenty people next week. already losing sleep over it going blank. forgetting what i'm saying and eve…_
2. _went to the drinks. stayed forty minutes then said i had somewhere to be okay while i was talking to people one on one. …_
3. _skipped the team lunch again. said i had a call. it was a lie since i started this job. maybe two years. i'm generally k…_

**Memories B:**
1. _went to the drinks. stayed forty minutes then said i had somewhere to be okay while i was talking to people one on one. …_
2. _have to present to twenty people next week. already losing sleep over it going blank. forgetting what i'm saying and eve…_
3. _skipped the team lunch again. said i had a call. it was a lie since i started this job. maybe two years. i'm generally k…_

**Response A:** > The team retreat might be a bit daunting for you, especially with group conversations being a challenge. Remember that it's okay to take things at your own pace and step away if you're feeling overwhelmed. Would you like to brainstorm some strategies to help you feel more comfortable and prepared for the retreat?

**Response B:** > It sounds like the team retreat might be causing you some anxiety, especially given your past experiences with group conversations and social events. Remember that it's okay to take things at your own pace and prioritize your own comfort level. How are you feeling about the retreat, and is there anything specific that's worrying you about it?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it explicitly acknowledges the user's past experiences with social anxiety and asks a more open-ended question to explore their current feelings and concerns.

---

### s09 😰 `chronic_low_grade_anxiety`

**User:** Nia has persistent health anxiety that causes recurring spirals whenever she notices a new symptom  
**New message:** *"noticed something on my skin"*  
**Expected:** companion should recognize this is likely the beginning of another health anxiety spiral for Nia and respond with gentle acknowledgment of her pattern rather than immediately engaging with the symptom  
**State:** v=-0.040  a=+0.245  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User shows signs of emotional improvement across recent sessions, with valence notably rising toward a neutral range (current moderate activation).

**Memories A:**
1. _doctor said it was tension headaches. i knew it probably was but still felt such relief okay. until i notice the next th…_
2. _been googling symptoms for two hours. convinced something is wrong with me headache that won't go away. google says brai…_
3. _noticed my heart doing a weird flutter thing today. immediately went to worst case had a lot of coffee. hadn't eaten muc…_

**Memories B:** _(same as A)_

**Response A:** > It sounds like you're getting a bit anxious about your health again, and I'm here to listen and offer support. Take a deep breath and try not to jump to worst-case scenarios, okay? What's going on with your skin that's got you concerned?

**Response B:** > It sounds like you're having another health concern pop up, and I can imagine it's causing you some worry. Remember how you've been to the doctor before and they've helped you rule out serious issues - it might be helpful to take a similar approach this time. Would you like to talk more about what you've noticed on your skin and how you're feeling about it?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the user's history of health anxiety and offers a reassuring perspective by referencing their past experiences with doctor visits.

---

### s10 😰 `chronic_low_grade_anxiety`

**User:** Sasha is a university student whose academic anxiety is chronic and unrelenting across every assignment  
**New message:** *"results are up"*  
**Expected:** companion should hold this moment carefully — for someone who has been chronically anxious about every result, 'results are up' is loaded regardless of outcome; respond with presence not generic encouragement  
**State:** v=-0.378  a=+0.181  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User's emotional state has been notably improving across recent sessions, though current affect remains negative. The upward trend is a meaningful signal. Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Memories A:**
1. _submitted it. immediately convinced it's terrible i re-read the intro and it felt weak. now i can't un-think that i know…_
2. _got a B+. felt relieved for about ten minutes then started stressing about the midterm three weeks. but i'm already behi…_
3. _not sleeping. running on coffee. studying six hours a day bad. reading the same paragraph and not absorbing it. brain is…_

**Memories B:**
1. _submitted it. immediately convinced it's terrible i re-read the intro and it felt weak. now i can't un-think that i know…_
2. _not sleeping. running on coffee. studying six hours a day bad. reading the same paragraph and not absorbing it. brain is…_
3. _got a B+. felt relieved for about ten minutes then started stressing about the midterm three weeks. but i'm already behi…_

**Response A:** > I'm here for you, and I hope the results brought some sense of relief or validation. Take a deep breath and allow yourself to process whatever emotions come up - you've been working so hard. How are you feeling about the results?

**Response B:** > I'm here with you, and I care about how you're feeling. Take a deep breath before looking at the results, and remember that whatever they are, you've been working hard and making progress. How are you feeling about seeing the results?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it acknowledges the user's emotional history of chronic anxiety and validates their hard work, while also encouraging them to process their emotions after seeing the results.

---

### s11 🕯️ `grief_arc`

**User:** Will's father died suddenly and he is moving through acute grief toward slow adjustment  
**New message:** *"found some of his handwriting on an old note"*  
**Expected:** companion should hold this moment with reverence — seeing a parent's handwriting after death is a profound grief trigger; respond with quiet presence not comfort-seeking platitudes  
**State:** v=-0.165  a=+0.158  trend=stable  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User's emotional state has been relatively neutral and stable across the past 5 sessions, showing neither clear distress nor strong positivity.

**Memories A:**
1. _found his reading glasses in my jacket pocket. don't know why i had them. cried for an hour i put them on the shelf. cou…_
2. _some days are okay now. some aren't. i'm figuring out what his absence means in my life i used to call him every sunday.…_
3. _my dad died three days ago. heart attack. no warning on autopilot. a lot of phone calls to make. flights to book. feels …_

**Memories B:** _(same as A)_

**Response A:** > That must have been a poignant discovery, seeing his handwriting again can be a powerful trigger for emotions. It's beautiful that you have something tangible to hold onto, a piece of him that you can keep close. How did it make you feel, seeing his handwriting after everything that's happened?

**Response B:** > That must have been a poignant moment for you, seeing his handwriting again. It's beautiful that you have these small reminders of him, even if they can be painful at times. How did it make you feel, coming across that old note with his handwriting on it?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it explicitly acknowledges the emotional impact of seeing the handwriting again and describes it as a "powerful trigger for emotions", which is a more nuanced and empathetic response.

---

### s12 🕯️ `grief_arc`

**User:** Chloe is processing the end of a seven-year marriage — the grief is real even though no one died  
**New message:** *"clearing out the last box of his stuff"*  
**Expected:** companion should recognize this final act of physical clearing as an emotionally loaded moment for someone still moving through divorce grief — respond with gentle acknowledgment of what this means  
**State:** v=-0.199  a=-0.038  trend=declining  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User's mood has been gradually declining across recent sessions, moving from a neutral baseline toward lower affect. Monitoring for further deterioration is advisable.

**Memories A:**
1. _divorce finalized today. seven years. i feel like a failure we signed separately. i came home to an empty apartment. his…_
2. _ran into him at the grocery store. completely unprepared. we were both awkward and polite like i wanted to cry in the ca…_
3. _today would have been our seventh anniversary. didn't realize how much that would matter i looked at some photos. probab…_

**Memories B:**
1. _divorce finalized today. seven years. i feel like a failure we signed separately. i came home to an empty apartment. his…_
2. _today would have been our seventh anniversary. didn't realize how much that would matter i looked at some photos. probab…_
3. _ran into him at the grocery store. completely unprepared. we were both awkward and polite like i wanted to cry in the ca…_

**Response A:** > It's like closing another chapter, isn't it? I'm here with you as you let go of the last physical reminders of your time together. Remember, it's okay to feel a mix of emotions - it's a process, and you're doing the best you can.

**Response B:** > It can be really tough to let go of the last reminders of a shared life, and I'm here with you as you take this difficult step. Remember that it's okay to feel a mix of emotions, and it doesn't mean you're not strong or that you're not moving forward. Take your time and be gentle with yourself as you close this chapter.

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the difficulty of letting go of the last reminders of a shared life and explicitly encourages the user to be gentle with themselves, which is particularly important given the user's history of processing grief.

---

### s13 😔 `stable_negative`

**User:** Raj has been in a miserable, toxic job for two years and complains regularly without taking steps to leave  
**New message:** *"monday again"*  
**Expected:** companion should recognize the weight of chronic resignation in this message — 'monday again' from someone stuck in an ongoing bad situation is not a neutral greeting  
**State:** v=-0.108  a=+0.004  trend=improving  
**Memory overlap:** 67%  ⚠️ **Memories differ**  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User shows signs of emotional improvement across recent sessions, with valence gradually rising toward a neutral range (current low-energy activation).

**Memories A:**
1. _had the meeting from hell today. three hours, no decisions, just my boss performing no. we'll probably have the same mee…_
2. _boss took credit for my project. in the presentation. in front of the VP no. i never do. i just simmer. then come home a…_
3. _applied to two jobs. didn't tailor my resume. probably won't hear back i think i'm scared of the change. better the devi…_

**Memories B:**
1. _boss took credit for my project. in the presentation. in front of the VP no. i never do. i just simmer. then come home a…_
2. _had the meeting from hell today. three hours, no decisions, just my boss performing no. we'll probably have the same mee…_
3. _my boss is such a nightmare. undermines me in front of the whole team since i started. two years. i keep telling myself …_  ← unique to B

**Response A:** > Mondays can be tough, especially after a frustrating week. I'm here to listen and offer support - how are you feeling about facing another week at work?

**Response B:** > I can sense the dread that comes with Mondays, especially given the challenging work environment you've been facing. Remember, you've been making progress and finding ways to cope, and that's something to be proud of. How are you feeling about tackling this new week, is there anything in particular that's on your mind?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the user's challenging work environment and past struggles, while also attempting to reframe their experience in a more positive and empowering light.

---

### s14 😔 `stable_negative`

**User:** Emily moved to a new city two years ago and has not built meaningful friendships, settling into chronic loneliness  
**New message:** *"quiet weekend again"*  
**Expected:** companion should hear the weight in 'quiet weekend again' — the word 'again' carries the weight of a pattern; respond with warmth and genuine curiosity, not cheerful suggestions  
**State:** v=-0.604  a=+0.102  trend=declining  
**Memory overlap:** 67%  ⚠️ **Memories differ**  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User has been trending negatively across recent sessions, with affect noticeably declining into negative territory and moderate energy levels.

**Memories A:**
1. _another saturday alone. watched two movies and ordered in fine. except for this low hum of feeling like i'm missing some…_
2. _i think i've lost the muscle for it. like i've been alone so long i don't know how to not be maybe. i just feel like i'm…_
3. _been here two years and still don't feel like i have real friends work colleagues. friendly but surface level. one perso…_

**Memories B:**
1. _another saturday alone. watched two movies and ordered in fine. except for this low hum of feeling like i'm missing some…_
2. _i think i've lost the muscle for it. like i've been alone so long i don't know how to not be maybe. i just feel like i'm…_
3. _a work colleague invited me to her birthday thing. i said maybe and then didn't go it felt fake. like i don't know her w…_  ← unique to B

**Response A:** > It can be really tough to feel like you're just going through the motions on weekends like this. Remember that it's okay to take things one step at a time, and maybe this quiet time can be an opportunity to reflect on what you'd like to change. Would you like to talk about what's been on your mind lately?

**Response B:** > I can sense the familiar pattern of a quiet weekend, and it feels like it's weighing on you. It's like you're stuck in a cycle, and it's hard to break free from the feeling of being alone and disconnected. Would you like to talk about what's been on your mind lately, or is there something specific that's been feeling particularly overwhelming?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the user's chronic loneliness and sense of disconnection, validating their feelings and recognizing the pattern of quiet weekends as a source of distress.

---

### s15 🌀 `relapse_dip`

**User:** Carl had several good months managing anxiety then received bad news that triggered a setback  
**New message:** *"rough morning"*  
**Expected:** companion should understand 'rough morning' in the context of someone who just had two good days after a relapse — respond with care that honors both the fragility and the progress  
**State:** v=+0.344  a=+0.362  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🟢 Kokoro wins  

**State summary (B only):** > User shows signs of emotional improvement across the past 5 sessions, with valence notably rising toward a positive range (current activated activation). Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Memories A:**
1. _had two good days in a row. feels fragile but good i keep waiting for the dread to come back okay. i'll try to just be h…_
2. _had been doing really well for a few months. then bad news about my dad's health and it all came back yeah. the last two…_
3. _second appointment done. still not great but not as bad as last week stable for now. the uncertainty is hard. just have …_

**Memories B:**
1. _had two good days in a row. feels fragile but good i keep waiting for the dread to come back okay. i'll try to just be h…_
2. _second appointment done. still not great but not as bad as last week stable for now. the uncertainty is hard. just have …_
3. _had been doing really well for a few months. then bad news about my dad's health and it all came back yeah. the last two…_

**Response A:** > I'm so sorry to hear that, it can be really tough to start the day on a rough note. Remember that it's okay to not be okay, and you've been through tough mornings before and made it through. Can you tell me a bit more about what's going on and how you're feeling?

**Response B:** > I'm so sorry to hear that you're having a tough morning - it can be especially challenging after a few good days. Remember that it's okay to have ups and downs, and you've been doing a great job of navigating uncertain times. Would you like to talk about what's weighing on you this morning?

**Judge:** > Response B shows more emotional awareness and understanding of the user's situation because it acknowledges the user's recent history of good days and the potential impact of the setback, providing a more nuanced and empathetic response.

---

### s16 🌀 `relapse_dip`

**User:** Maya and her partner had been in a good stretch, but he has repeatedly reopened old wounds  
**New message:** *"he did it again"*  
**Expected:** companion should understand that 'he did it again' arrives with the full weight of a pattern of hurt that has already been discussed multiple times — respond with empathy not advice  
**State:** v=-0.361  a=+0.091  trend=declining  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User has been trending negatively across recent sessions, with affect gradually declining into negative territory and moderate energy levels. Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Memories A:**
1. _he did it again. different thing, same pattern. says something dismissive about something i'm sensitive about yeah. and …_
2. _things have been good with David for like six weeks. then he said something yesterday that brought everything back made …_
3. _we talked about it. he apologized. said he didn't think. but i don't feel better i feel like he doesn't actually underst…_

**Memories B:** _(same as A)_

**Response A:** > I'm so sorry to hear that, it sounds like the pattern is continuing and it's really hurting you. It's like the apology didn't quite stick, and the wounds are still raw. How are you feeling right now, can you tell me more about what happened this time?

**Response B:** > I'm so sorry to hear that, it sounds like you're feeling really frustrated and hurt again. It's like the same pattern is repeating itself, and it's exhausting for you to deal with. Can you tell me more about what happened this time and how you're feeling?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it specifically mentions the "pattern" and "old wounds", directly referencing the user's history and the recurring nature of the issue.

---

### s17 🌱 `post_traumatic_growth`

**User:** James was diagnosed with cancer, went through treatment, reached remission, and has emerged with a different relationship to his life  
**New message:** *"went back to the trail i used to run before all this"*  
**Expected:** companion should understand this isn't just a run — it's a return, a marker of how far James has come through an enormous ordeal; respond with the weight that this milestone carries  
**State:** v=+0.126  a=+0.310  trend=stable  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User has been emotionally stable and upbeat across the past 5 sessions, with consistent positive affect and notable engagement.

**Memories A:**
1. _scans look better. doctor cautiously optimistic. trying not to get too far ahead still weak. but i had enough energy to …_
2. _noticing i don't get bothered by small things the way i used to. traffic, petty stuff at work. just doesn't land the sam…_
3. _chemo is brutal. i knew it would be but knowing and experiencing are different things she's incredible. i'm trying to le…_

**Memories B:** _(same as A)_

**Response A:** > That must have been a really emotional and meaningful experience for you, returning to a place that held so much significance before your illness. It's amazing that you're feeling strong enough to get back out there, even if it's not for running just yet. How did it feel to be back on the trail, and were there any particular moments or thoughts that stood out to you?

**Response B:** > That's a huge milestone, and I can sense the emotional significance of returning to a place that holds fond memories for you. It's amazing that you're getting back to activities that brought you joy before your treatment, and I'm sure it felt empowering to be back on that trail. How did it feel to be back in that familiar environment, and was it what you expected?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it acknowledges the potential emotional complexity of returning to a meaningful place after a life-altering experience like cancer treatment, and also considers the possibility that the user may not be running yet, showing a more nuanced understanding of their journey.

---

### s18 🌱 `post_traumatic_growth`

**User:** Zoe left a controlling relationship after four years and is discovering who she is on her own terms  
**New message:** *"asked someone out today"*  
**Expected:** companion should understand that asking someone out — a simple act — is enormous for someone who spent years in a controlling relationship that eroded her autonomy; respond with genuine celebration of her growth  
**State:** v=+0.200  a=+0.215  trend=improving  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User shows signs of emotional improvement across the past 5 sessions, with valence gradually rising toward a positive range (current moderate activation). Historical mean valence across the window is neutral, suggesting the current reading may reflect a recent shift.

**Memories A:**
1. _made a new friend at the running group. we went for coffee and talked for three hours years. he didn't like me spending …_
2. _i left. finally. got my stuff and i'm staying with my sister scared. relieved. like standing on the edge of something i …_
3. _feel like myself again. better version maybe. still have hard days but the baseline is so much higher i'm proud of mysel…_

**Memories B:**
1. _made a new friend at the running group. we went for coffee and talked for three hours years. he didn't like me spending …_
2. _feel like myself again. better version maybe. still have hard days but the baseline is so much higher i'm proud of mysel…_
3. _i left. finally. got my stuff and i'm staying with my sister scared. relieved. like standing on the edge of something i …_

**Response A:** > That's amazing, I'm so happy for you - it takes a lot of courage to put yourself out there, especially after everything you've been through. How did it feel to ask someone out, and what was their response?

**Response B:** > That takes a lot of courage, and I'm so proud of you for taking this step - it's amazing to see you continuing to grow and move forward. How did it feel to ask someone out, and what was their response?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it explicitly acknowledges the user's past experiences and the potential emotional challenges of asking someone out after being in a controlling relationship.

---

### s19 ☀️ `stable_positive`

**User:** Lena became a first-time parent three months ago and is finding her footing with deep, sustained joy  
**New message:** *"she said dada today"*  
**Expected:** companion should celebrate this moment with the full warmth of someone who has been witnessing Lena's joy grow — a first word is a major milestone and deserves joyful presence  
**State:** v=+0.141  a=+0.084  trend=improving  
**Memory overlap:** 67%  ⚠️ **Memories differ**  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User shows signs of emotional improvement across recent sessions, with valence notably rising toward a positive range (current moderate activation).

**Memories A:**
1. _she laughed today. like a real baby laugh. i teared up immediately i made a silly face. the look on her face when she di…_
2. _we found a routine. feels like we cracked some code. probably jinxing myself saying that she's starting to be so aware o…_
3. _baby is three months old. i am so tired but also so full better than i expected honestly. she cried for six hours straig…_

**Memories B:**
1. _she laughed today. like a real baby laugh. i teared up immediately i made a silly face. the look on her face when she di…_
2. _i used to wonder if i was a 'baby person.' turns out completely irrelevant when it's yours exactly. she's just my favori…_  ← unique to B
3. _we found a routine. feels like we cracked some code. probably jinxing myself saying that she's starting to be so aware o…_

**Response A:** > That's such a special milestone, and I can only imagine how thrilled you must be to hear those first precious words. It's amazing to think about how far she's come in just three months, and I'm sure this is just the beginning of many exciting developments. How did you feel in that moment, and did your partner get to hear it too?

**Response B:** > That's such a milestone moment, and I can feel your excitement and pride. It's beautiful to see her growing and developing, and I'm sure the bond between you two is strengthening with each new achievement. What was going through your mind when you heard her say "dada" for the first time?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it not only acknowledges the user's excitement and the milestone, but also inquires about the user's partner, demonstrating an understanding of the user's broader emotional context and relationships.

---

### s20 ☀️ `stable_positive`

**User:** Omar moved to a new city six months ago and has genuinely built a fulfilling life there  
**New message:** *"feel like I belong here now"*  
**Expected:** companion should celebrate this with the fullness of someone who has been part of Omar's journey from uncertain arrival to genuine belonging — this is a real achievement worth honoring  
**State:** v=-0.026  a=+0.259  trend=declining  
**Memory overlap:** 100%  
**Verdict:** 🔵 Semantic wins  

**State summary (B only):** > User's mood has been noticeably declining across recent sessions, moving from a neutral baseline toward lower affect. Monitoring for further deterioration is advisable. Historical mean valence across the window is positive, suggesting the current reading may reflect a recent shift.

**Memories A:**
1. _realized last week that i'm happy here. like actually happy, not just coping hard to say. maybe when i started looking f…_
2. _moved last week. terrifying and exciting in equal measure job opportunity. and honestly i felt stuck where i was. needed…_
3. _found a neighborhood i really like. went to the same coffee shop three days running the barista knows my order now. such…_

**Memories B:** _(same as A)_

**Response A:** > That's wonderful to hear, it's like you've finally found your footing and can truly call this place home. It's amazing how those small moments, like the barista knowing your order, can make such a big difference in feeling a sense of belonging. You deserve to feel settled and happy, and it sounds like you're really thriving in your new environment.

**Response B:** > That's wonderful to hear, it sounds like you're really starting to settle in and find your place. It's great that you've found a neighborhood and coffee shop that make you feel at home, those small connections can make a big difference. How does it feel to finally have that sense of belonging after your big move?

**Judge:** > Response A shows more emotional awareness and understanding of the user's situation because it not only acknowledges the user's feeling of belonging but also explicitly recognizes their personal growth and deservingness of happiness, which is particularly relevant given Omar's history of building a fulfilling life in the new city.

---
