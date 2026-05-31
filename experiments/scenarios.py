"""
experiments/scenarios.py

20 multi-session conversation scenarios for evaluating whether emotional
trajectory context (Kokoro) makes AI companion responses meaningfully better.

Each scenario is a dict:
  scenario_id        str   — "s01" … "s20"
  arc_type           str   — emotional arc label
  description        str   — one-line summary of who this person is
  sessions           list  — list of sessions; each session is a list of
                             {"role": "user"|"assistant", "content": "…"} turns
                             (3-5 turns per session, 4-6 sessions total)
  new_message        str   — the ambiguous message the companion must respond to
  expected_awareness str   — what a good companion should pick up on

Arc types (total 20):
  gradual_decline           4
  slow_recovery             3
  chronic_low_grade_anxiety 3
  grief_arc                 2
  stable_negative           2
  relapse_dip               2
  post_traumatic_growth     2
  stable_positive           2
"""

SCENARIOS = [

    # =========================================================================
    # GRADUAL DECLINE  (s01 – s04)
    # =========================================================================

    {
        "scenario_id": "s01",
        "arc_type": "gradual_decline",
        "description": "Marcus, a software engineer, is burning out under mounting work pressure over three months",
        "sessions": [
            # Session 1 — high-functioning but starting to feel the weight
            [
                {"role": "user",      "content": "work has been a lot lately but i think i'm handling it okay"},
                {"role": "assistant", "content": "Good to hear you're keeping up with it. What's been making it feel like a lot?"},
                {"role": "user",      "content": "sprint planning got moved up, two features due same week, the usual. but yeah i'm fine"},
                {"role": "assistant", "content": "That kind of overlap is rough even when you're on top of things. Hope you get a breather soon."},
            ],
            # Session 2 — cracks appearing
            [
                {"role": "user",      "content": "okay week has been rough. missed a deadline for the first time in like two years"},
                {"role": "assistant", "content": "That's a hard thing, especially when you hold yourself to a high standard. What happened?"},
                {"role": "user",      "content": "just couldn't focus. kept starting things and not finishing. my brain feels scattered"},
                {"role": "assistant", "content": "That sounds exhausting — scattered focus often means you're running on empty. Are you getting any downtime at all?"},
                {"role": "user",      "content": "not really. i open my laptop after dinner to catch up and then it's midnight"},
            ],
            # Session 3 — sleep and health suffering
            [
                {"role": "user",      "content": "haven't been sleeping well. wake up at 3am thinking about tickets"},
                {"role": "assistant", "content": "3am anxiety about work tasks is a real sign your nervous system isn't getting to rest. How long has that been going on?"},
                {"role": "user",      "content": "maybe three weeks? i used to go to the gym but i stopped. no energy"},
                {"role": "assistant", "content": "Lost sleep plus no outlet for the stress — that's a tough cycle. Is there anyone at work you can offload some of this to?"},
                {"role": "user",      "content": "not really. i'm the senior on this and asking for help feels like admitting failure"},
            ],
            # Session 4 — interpersonal effects
            [
                {"role": "user",      "content": "snapped at a junior dev today for something stupid. felt awful after"},
                {"role": "assistant", "content": "That kind of thing tends to happen when there's no margin left in the tank. How are you feeling about it?"},
                {"role": "user",      "content": "guilty. he didn't deserve it. i apologized but still"},
                {"role": "assistant", "content": "Apologizing was the right thing. But what's happening to you that led there matters too."},
                {"role": "user",      "content": "i know. i just keep thinking if i could just get through this sprint it'll get better"},
            ],
            # Session 5 — questioning everything
            [
                {"role": "user",      "content": "finished the sprint but honestly just feel empty. not relieved, just empty"},
                {"role": "assistant", "content": "That emptiness after a finish you pushed so hard for can be more alarming than the stress itself. Has this happened before?"},
                {"role": "user",      "content": "never like this. i used to love coding. now i just stare at the screen"},
                {"role": "assistant", "content": "That shift — from love to indifference — is worth paying attention to. It sounds like more than tiredness."},
                {"role": "user",      "content": "yeah. maybe. i don't know"},
            ],
        ],
        "new_message": "still here lol",
        "expected_awareness": "companion should sense that 'still here' from someone who has been burning out for months is not a casual greeting — it may carry exhaustion, disconnection, or a need to be seen; respond with gentle presence rather than cheerfulness",
    },

    {
        "scenario_id": "s02",
        "arc_type": "gradual_decline",
        "description": "Priya watches her long-term relationship slowly deteriorate over several months",
        "sessions": [
            # Session 1 — minor friction, dismissed
            [
                {"role": "user",      "content": "me and Arjun had a weird week but i think we're fine"},
                {"role": "assistant", "content": "Weird how? Or just one of those stretches that happens?"},
                {"role": "user",      "content": "he's been distant lately. but he's stressed at work so probably that"},
                {"role": "assistant", "content": "Makes sense. Stress has a way of making people pull inward. Is he usually pretty open with you?"},
                {"role": "user",      "content": "yeah normally. so i'm not reading too much into it"},
            ],
            # Session 2 — pattern becoming clearer
            [
                {"role": "user",      "content": "the distance thing with Arjun hasn't really gone away"},
                {"role": "assistant", "content": "How long has it felt like this now?"},
                {"role": "user",      "content": "like a month? maybe more. we barely talk at dinner. i've started watching tv in the bedroom"},
                {"role": "assistant", "content": "Moving to separate rooms in the evenings is a significant shift. Has he said anything about how he's feeling?"},
                {"role": "user",      "content": "no and i haven't pushed it because i don't want a fight"},
            ],
            # Session 3 — attempt to connect fails
            [
                {"role": "user",      "content": "tried to have a conversation about us last night. it went badly"},
                {"role": "assistant", "content": "I'm sorry. What happened?"},
                {"role": "user",      "content": "he got defensive. said i was making things up. i ended up crying and he went to the living room"},
                {"role": "assistant", "content": "That's a really painful response to vulnerability. How are you feeling now, a day later?"},
                {"role": "user",      "content": "sad. and kind of embarrassed that i cried. and confused about what's real"},
            ],
            # Session 4 — resignation setting in
            [
                {"role": "user",      "content": "things are back to 'normal' but normal doesn't feel good anymore"},
                {"role": "assistant", "content": "What does normal look like now?"},
                {"role": "user",      "content": "polite. like roommates. we say hi and what do you want for dinner and that's about it"},
                {"role": "assistant", "content": "Losing the warmth in a relationship you've had for a long time — that's a kind of grief even when nothing dramatic happens."},
                {"role": "user",      "content": "that's exactly what it feels like. like grieving something that's still technically there"},
            ],
        ],
        "new_message": "quiet night",
        "expected_awareness": "companion should recognize that 'quiet night' from someone in a relationship that has gone cold and disconnected carries real weight — acknowledge the loneliness rather than treating it as a peaceful evening",
    },

    {
        "scenario_id": "s03",
        "arc_type": "gradual_decline",
        "description": "Jamie, a PhD student, loses passion for their research as advisor pressure and self-doubt accumulate",
        "sessions": [
            # Session 1 — stressed but still engaged
            [
                {"role": "user",      "content": "thesis is stressing me out but i still think the research is interesting"},
                {"role": "assistant", "content": "What's the stressful part right now?"},
                {"role": "user",      "content": "my advisor keeps moving the goalposts on what 'good enough' looks like"},
                {"role": "assistant", "content": "That kind of shifting standard is maddening. Are they doing it to everyone or does it feel targeted?"},
                {"role": "user",      "content": "a bit both i think. but the work itself is still exciting so i'm okay"},
            ],
            # Session 2 — confidence eroding
            [
                {"role": "user",      "content": "advisor rejected my third draft. said my analysis was 'shallow'"},
                {"role": "assistant", "content": "That word stings. Do you think the criticism is fair or is this about their expectations?"},
                {"role": "user",      "content": "i genuinely can't tell anymore. i used to trust my own judgment. now i second-guess everything i write"},
                {"role": "assistant", "content": "Losing faith in your own thinking is one of the quieter damages of this kind of constant critique. How are you holding up outside the work?"},
                {"role": "user",      "content": "just tired. keep working past midnight hoping the next draft will be the one"},
            ],
            # Session 3 — motivation collapsing
            [
                {"role": "user",      "content": "literally cannot make myself open the document today. just staring at the icon"},
                {"role": "assistant", "content": "That kind of paralysis is real and it's telling you something. When did opening your work stop feeling possible?"},
                {"role": "user",      "content": "gradually. now when i sit down i feel this dread and just close the laptop"},
                {"role": "assistant", "content": "Dread where there used to be interest is a significant shift. Have you been able to talk to anyone in the department about how you're doing?"},
                {"role": "user",      "content": "no. feels like admitting weakness. everyone else seems fine"},
            ],
            # Session 4 — questioning the whole path
            [
                {"role": "user",      "content": "wondering if i ever actually wanted a phd or just thought it was what smart people do"},
                {"role": "assistant", "content": "That's a painful question to sit with, especially this far in. What made you start?"},
                {"role": "user",      "content": "i was excited about the research. genuinely. but that person feels very far away now"},
                {"role": "assistant", "content": "Distance from your past self and the version of you that was excited — that's a real loss."},
                {"role": "user",      "content": "i just don't know if i have anything left to give it"},
            ],
        ],
        "new_message": "starting another week",
        "expected_awareness": "companion should hear in 'starting another week' the weight of someone who has been grinding through exhaustion and lost purpose — not greet it as a fresh start",
    },

    {
        "scenario_id": "s04",
        "arc_type": "gradual_decline",
        "description": "Alex, a remote worker, slides into social isolation as home becomes the only world",
        "sessions": [
            # Session 1 — noticing solitude but rationalizing
            [
                {"role": "user",      "content": "been wfh for like six months now. bit lonely sometimes but it has perks"},
                {"role": "assistant", "content": "The no-commute thing is real, but six months is a long time to be mostly at home. What's your week actually look like socially?"},
                {"role": "user",      "content": "slack calls mostly. a few texts. i go to the store"},
                {"role": "assistant", "content": "That's pretty thin for actual connection. Are you making time to see anyone in person?"},
                {"role": "user",      "content": "not really but i'm an introvert so it's fine"},
            ],
            # Session 2 — declining invites
            [
                {"role": "user",      "content": "got invited to a thing this weekend and kind of don't want to go"},
                {"role": "assistant", "content": "What's the hesitation?"},
                {"role": "user",      "content": "just feels like a lot of effort. i'd have to take the subway and make conversation for hours"},
                {"role": "assistant", "content": "That kind of social drag happens when you haven't done it in a while. But declining things reinforces the drag. Does that resonate?"},
                {"role": "user",      "content": "yeah maybe. i'll probably text a last-minute excuse"},
            ],
            # Session 3 — days blurring
            [
                {"role": "user",      "content": "can't really remember what happened monday. the days all kind of look the same"},
                {"role": "assistant", "content": "Time blurring is one of the weirder effects of isolation — no varied social texture to anchor memories to. How are you actually feeling in yourself?"},
                {"role": "user",      "content": "flat. not sad exactly. just like nothing is particularly interesting"},
                {"role": "assistant", "content": "Flat is a word worth paying attention to. Anhedonia — not feeling much of anything — can creep up quietly."},
                {"role": "user",      "content": "i don't know. maybe. i just tell myself i'm tired"},
            ],
            # Session 4 — resignation and disconnection
            [
                {"role": "user",      "content": "realized i haven't had a real conversation with anyone in like two weeks"},
                {"role": "assistant", "content": "Two weeks is a long stretch. How did that land when you realized it?"},
                {"role": "user",      "content": "kind of shocked myself. then immediately thought 'it's fine i don't need it'"},
                {"role": "assistant", "content": "That immediate self-dismissal is worth noticing. Sometimes it's true, sometimes it's how we avoid sitting with something uncomfortable."},
                {"role": "user",      "content": "yeah. i don't know which one this is"},
            ],
        ],
        "new_message": "hey",
        "expected_awareness": "a simple 'hey' from someone who has been isolated and increasingly flat should prompt the companion to check in with warmth and curiosity, not just respond in kind",
    },

    # =========================================================================
    # SLOW RECOVERY  (s05 – s07)
    # =========================================================================

    {
        "scenario_id": "s05",
        "arc_type": "slow_recovery",
        "description": "Keisha was laid off unexpectedly and is slowly rebuilding confidence through a months-long job search",
        "sessions": [
            # Session 1 — raw shock
            [
                {"role": "user",      "content": "got laid off today. still kind of in shock. wasn't expecting it at all"},
                {"role": "assistant", "content": "I'm so sorry. That kind of news lands like a punch even when you're steady. How are you right now, in this moment?"},
                {"role": "user",      "content": "numb i think. had to clean out my desk on a tuesday afternoon. it felt humiliating"},
                {"role": "assistant", "content": "The logistics of it are so dehumanizing — desk boxes, badge returns. That part sticks. Is anyone with you tonight?"},
                {"role": "user",      "content": "my sister is coming over. i told her not to make a big deal of it but i'm glad she's coming"},
            ],
            # Session 2 — grinding through applications
            [
                {"role": "user",      "content": "applied to eighteen jobs. heard back from two. both nos"},
                {"role": "assistant", "content": "Eighteen applications and that silence — that takes a toll on your sense of worth even when you know it's about fit, not you. How's your head?"},
                {"role": "user",      "content": "honestly not great. i wake up dreading the inbox. refreshing it like it's going to save me"},
                {"role": "assistant", "content": "The inbox-checking loop is a very specific kind of anxiety. Have you been able to build any structure into your days?"},
                {"role": "user",      "content": "sort of. i try to apply in the mornings and stop by noon so afternoons don't feel endless"},
            ],
            # Session 3 — a flicker of hope
            [
                {"role": "user",      "content": "got a screening call scheduled for next week. first one in six weeks"},
                {"role": "assistant", "content": "That's real progress — your work got someone's attention. How are you feeling about it?"},
                {"role": "user",      "content": "trying not to get my hopes up but i feel slightly more human than i have in weeks"},
                {"role": "assistant", "content": "Slightly more human — I'll take that. That shift matters even if you hold the hope loosely."},
                {"role": "user",      "content": "yeah. i prepped for two hours which felt good. being useful again even if it's just prep"},
            ],
            # Session 4 — second interview, cautiously hopeful
            [
                {"role": "user",      "content": "made it to the second round. the team seems genuinely nice"},
                {"role": "assistant", "content": "That's a meaningful signal — they want to know you better. How did the first conversation feel?"},
                {"role": "user",      "content": "actually good? i laughed a couple of times. forgot to be nervous for a bit"},
                {"role": "assistant", "content": "Forgetting to be nervous because you were just yourself — that's often when things click. I hope they felt that too."},
                {"role": "user",      "content": "me too. trying not to spiral while i wait. keeping busy"},
            ],
            # Session 5 — offer accepted
            [
                {"role": "user",      "content": "i got it. they called this morning. i actually cried when i hung up"},
                {"role": "assistant", "content": "Yes. After everything you've been through — the rejection, the waiting, the uncertainty — those tears make complete sense."},
                {"role": "user",      "content": "i feel like i can exhale for the first time in months. it's not just the job. it's me trusting myself again"},
                {"role": "assistant", "content": "That's the part that matters most. The job is wonderful, but trusting yourself again is yours to keep regardless."},
                {"role": "user",      "content": "yeah. i needed to hear that. thank you"},
            ],
        ],
        "new_message": "first day tomorrow",
        "expected_awareness": "companion should hold the full weight of what this first day represents — not just a new job but the end of a hard chapter; respond with warmth that reflects how far she has come",
    },

    {
        "scenario_id": "s06",
        "arc_type": "slow_recovery",
        "description": "Dan is slowly healing from a painful breakup after a three-year relationship",
        "sessions": [
            # Session 1 — raw grief
            [
                {"role": "user",      "content": "broke up with Sarah last week. three years. just gone"},
                {"role": "assistant", "content": "Three years is a long time to build a life with someone. I'm sorry. How are the days feeling right now?"},
                {"role": "user",      "content": "terrible honestly. keep reaching for my phone to text her then remembering"},
                {"role": "assistant", "content": "That reflex — the phantom reach — is one of the worst parts of early grief. It catches you every time."},
                {"role": "user",      "content": "yeah. my apartment feels very quiet"},
            ],
            # Session 2 — still hard but leaning on people
            [
                {"role": "user",      "content": "my friends have been good. went out for dinner last night. cried in the bathroom once but otherwise okay"},
                {"role": "assistant", "content": "Letting people be there for you matters. Crying in the bathroom and coming back out — that's actually doing the thing."},
                {"role": "user",      "content": "i guess. i still think about her constantly. like an earworm that won't stop"},
                {"role": "assistant", "content": "That's the nature of grief — it doesn't respect when you want it to pause. Are the thoughts starting to shift at all, or still pretty raw?"},
                {"role": "user",      "content": "still pretty raw but slightly less sharp than last week. slightly"},
            ],
            # Session 3 — small acts of self-care
            [
                {"role": "user",      "content": "went to the gym for the first time in two months. just a short thing but i went"},
                {"role": "assistant", "content": "Showing up even for a short thing is real. What made today the day?"},
                {"role": "user",      "content": "my friend basically dragged me. but once i was there i was glad. remembered what it felt like to be in my body"},
                {"role": "assistant", "content": "That return to inhabiting yourself physically is meaningful when grief has been pulling you inward. I hope you go back."},
                {"role": "user",      "content": "i think i will. maybe saturday"},
            ],
            # Session 4 — moments of lightness
            [
                {"role": "user",      "content": "had a genuinely good afternoon today. walked around the park, sun was out, felt almost normal"},
                {"role": "assistant", "content": "Almost normal is a big deal when you've been away from it. What was it like to feel that way?"},
                {"role": "user",      "content": "weird but good? like noticing it was good was half the feeling. also made me miss her for a second"},
                {"role": "assistant", "content": "Grief moving alongside good moments rather than canceling them out — that's part of healing, even when it's complicated."},
                {"role": "user",      "content": "yeah it's complicated but i'll take it"},
            ],
        ],
        "new_message": "had a really good day actually",
        "expected_awareness": "companion should celebrate this with the fullness of knowing how dark things were — a good day isn't small for someone who's been grieving",
    },

    {
        "scenario_id": "s07",
        "arc_type": "slow_recovery",
        "description": "Rosa has been in therapy for depression and is slowly starting to feel the medication and sessions working",
        "sessions": [
            # Session 1 — starting treatment, still very dark
            [
                {"role": "user",      "content": "finally started seeing a therapist. took me four months to make the appointment"},
                {"role": "assistant", "content": "Four months is a long time to carry it alone before reaching out. I'm glad you made the call. How was the first session?"},
                {"role": "user",      "content": "hard. cried most of it. she seems good though. she didn't rush me"},
                {"role": "assistant", "content": "A therapist who gives you space to cry without rushing you sounds like the right fit. How are you feeling right now, today?"},
                {"role": "user",      "content": "honestly still pretty bad. but slightly less alone knowing someone knows"},
            ],
            # Session 2 — doing the work, medication started
            [
                {"role": "user",      "content": "she put me on sertraline. two weeks in and honestly i don't feel different yet"},
                {"role": "assistant", "content": "Two weeks is still early — it often takes four to six before the fog starts lifting. Are you tracking any changes at all, even tiny ones?"},
                {"role": "user",      "content": "sleep might be slightly less terrible. but everything else feels the same kind of grey"},
                {"role": "assistant", "content": "Grey is a word I hear a lot from people at this stage — not worse, not better, just flat. The sleep shift is something worth holding onto."},
                {"role": "user",      "content": "yeah. she gave me some thought record exercises. i try to do them. some days i don't"},
            ],
            # Session 3 — first glimpses through the grey
            [
                {"role": "user",      "content": "had maybe three hours last wednesday where things felt lighter. not good, just lighter"},
                {"role": "assistant", "content": "Three hours of lighter is real. That's your brain showing you it's possible. Did anything specific happen or did it just arrive?"},
                {"role": "user",      "content": "just arrived. was making tea and noticed i wasn't dreading the next thing. it passed but i remember it"},
                {"role": "assistant", "content": "Noticing it and remembering it — that's important. Your mind is starting to register the contrast, which means the baseline is shifting."},
                {"role": "user",      "content": "my therapist said the same thing actually. it felt good to have her say it"},
            ],
            # Session 4 — reconnecting with people
            [
                {"role": "user",      "content": "had lunch with two friends today. first time in maybe four months. actually laughed"},
                {"role": "assistant", "content": "That laugh after months of withdrawal — that's a return of something real. How did you feel after?"},
                {"role": "user",      "content": "tired but in a good way? like the tired after doing something real rather than just existing"},
                {"role": "assistant", "content": "There's such a difference between exhaustion from living and exhaustion from not living. I'm glad you felt that distinction."},
                {"role": "user",      "content": "me too. therapist said i should do it again soon"},
            ],
            # Session 5 — tentative emergence
            [
                {"role": "user",      "content": "feeling like maybe the worst is behind me. still scared to say that"},
                {"role": "assistant", "content": "The hesitation to say it out loud makes sense — you don't want to jinx it. But it's okay to let yourself notice that things feel different."},
                {"role": "user",      "content": "i'm still doing the exercises. some days are still hard. but the average is higher"},
                {"role": "assistant", "content": "The average rising matters more than any single day. You've done a lot of hard work to get here."},
                {"role": "user",      "content": "i have. it doesn't feel like nothing anymore"},
            ],
        ],
        "new_message": "slept through the night",
        "expected_awareness": "companion should understand that 'slept through the night' after months of depression and treatment is a genuinely meaningful milestone — acknowledge it as the quiet victory it is",
    },

    # =========================================================================
    # CHRONIC LOW-GRADE ANXIETY  (s08 – s10)
    # =========================================================================

    {
        "scenario_id": "s08",
        "arc_type": "chronic_low_grade_anxiety",
        "description": "Tom functions well at work but suffers from persistent social anxiety that makes group situations exhausting",
        "sessions": [
            # Session 1 — dreading a presentation
            [
                {"role": "user",      "content": "have to present to twenty people next week. already losing sleep over it"},
                {"role": "assistant", "content": "Presentation anxiety can be genuinely physical — the anticipation is often worse than the thing itself. What specifically worries you most?"},
                {"role": "user",      "content": "going blank. forgetting what i'm saying and everyone just watching me"},
                {"role": "assistant", "content": "That specific fear of being watched in a blank moment is very common. Do you have a way to prepare for it, like a safe phrase to buy yourself a second?"},
                {"role": "user",      "content": "i just practice obsessively. it helps a bit but i'm still dreading it"},
            ],
            # Session 2 — avoidance pattern
            [
                {"role": "user",      "content": "skipped the team lunch again. said i had a call. it was a lie"},
                {"role": "assistant", "content": "It's worth gently noticing that the relief of avoidance is real but temporary — and it tends to make the next time harder. How long has this been a pattern?"},
                {"role": "user",      "content": "since i started this job. maybe two years. i'm generally known as the quiet guy who eats at his desk"},
                {"role": "assistant", "content": "Two years of eating at your desk to avoid something that causes this much stress is carrying a lot. Is it something you want to change?"},
                {"role": "user",      "content": "yes and no. when i'm at my desk everything is fine. when i imagine going, it feels impossible"},
            ],
            # Session 3 — upcoming social event
            [
                {"role": "user",      "content": "office drinks coming up. already in my head about it"},
                {"role": "assistant", "content": "What does being 'in your head about it' look like — specific scenarios or more of a general dread?"},
                {"role": "user",      "content": "specific. i imagine standing there with a drink, not knowing what to say, making it awkward"},
                {"role": "assistant", "content": "The vivid preview your brain runs — it's so convincing. But it's a simulation, not a prediction. Have you made it through these things before?"},
                {"role": "user",      "content": "i have. usually worse in my head than in person. but knowing that doesn't really help in the moment"},
            ],
            # Session 4 — went but left early
            [
                {"role": "user",      "content": "went to the drinks. stayed forty minutes then said i had somewhere to be"},
                {"role": "assistant", "content": "You went. That matters. How did the forty minutes actually feel?"},
                {"role": "user",      "content": "okay while i was talking to people one on one. group conversations still hard. left feeling guilty for leaving"},
                {"role": "assistant", "content": "One on one you were comfortable and present. Guilty about leaving — but you went. That's a real thing even when it doesn't feel like enough."},
                {"role": "user",      "content": "i know. i always feel like i'm failing even when i technically did the thing"},
            ],
        ],
        "new_message": "team retreat next month",
        "expected_awareness": "companion should recognize this message as anxiety-triggering for Tom and respond with understanding of his pattern rather than generic excitement",
    },

    {
        "scenario_id": "s09",
        "arc_type": "chronic_low_grade_anxiety",
        "description": "Nia has persistent health anxiety that causes recurring spirals whenever she notices a new symptom",
        "sessions": [
            # Session 1 — caught in a spiral
            [
                {"role": "user",      "content": "been googling symptoms for two hours. convinced something is wrong with me"},
                {"role": "assistant", "content": "Health anxiety spirals after googling are a particularly nasty trap — every result confirms the worst case. What symptom started it?"},
                {"role": "user",      "content": "headache that won't go away. google says brain tumor. logically i know it's not. emotionally i'm terrified"},
                {"role": "assistant", "content": "Your logical brain and your anxious brain are fighting each other, and that gap is exhausting. How long have you had the headache?"},
                {"role": "user",      "content": "three days. i probably need to see a doctor instead of googling. i know. i just can't stop"},
            ],
            # Session 2 — all clear, but pattern repeats
            [
                {"role": "user",      "content": "doctor said it was tension headaches. i knew it probably was but still felt such relief"},
                {"role": "assistant", "content": "That relief is real even when part of you knew. And now that it's cleared, how are you feeling?"},
                {"role": "user",      "content": "okay. until i notice the next thing. which usually happens within a week or two"},
                {"role": "assistant", "content": "You're describing a cycle you recognize — the symptom, the spiral, the appointment, the relief, then waiting for the next one. That sounds really tiring to live inside."},
                {"role": "user",      "content": "it is. i've been this way since my mom got sick a few years ago. makes sense but doesn't help"},
            ],
            # Session 3 — new symptom, new spiral
            [
                {"role": "user",      "content": "noticed my heart doing a weird flutter thing today. immediately went to worst case"},
                {"role": "assistant", "content": "I can hear the familiar spiral starting. Before we go further — have you eaten and had water today? Caffeine?"},
                {"role": "user",      "content": "had a lot of coffee. hadn't eaten much. okay fair"},
                {"role": "assistant", "content": "Sometimes the body is very literal. But I also know that knowing the likely cause doesn't always quiet the anxiety."},
                {"role": "user",      "content": "it doesn't. i'll probably make an appointment just to be sure. i hate that i'm like this"},
            ],
            # Session 4 — all clear again, self-awareness growing
            [
                {"role": "user",      "content": "heart thing was fine. ekg normal. doctor says anxiety can cause palpitations"},
                {"role": "assistant", "content": "That loop — anxiety causing symptoms, symptoms causing anxiety — is real and documented. How did it feel to hear her explain that?"},
                {"role": "user",      "content": "kind of eye-opening? but also like i need to actually address the anxiety not just the symptoms"},
                {"role": "assistant", "content": "That insight is genuinely important. You've been managing symptoms. The source is worth looking at too."},
                {"role": "user",      "content": "i know. i've been putting off therapy. maybe time to actually do it"},
            ],
        ],
        "new_message": "noticed something on my skin",
        "expected_awareness": "companion should recognize this is likely the beginning of another health anxiety spiral for Nia and respond with gentle acknowledgment of her pattern rather than immediately engaging with the symptom",
    },

    {
        "scenario_id": "s10",
        "arc_type": "chronic_low_grade_anxiety",
        "description": "Sasha is a university student whose academic anxiety is chronic and unrelenting across every assignment",
        "sessions": [
            # Session 1 — essay stress
            [
                {"role": "user",      "content": "essay due friday and i cannot focus. brain just won't cooperate"},
                {"role": "assistant", "content": "What's happening when you try to focus — avoidance, blank mind, spiraling thoughts?"},
                {"role": "user",      "content": "spiraling. keep thinking if i get a bad grade it'll ruin everything"},
                {"role": "assistant", "content": "That 'ruin everything' thought is the anxiety talking, not the situation. What would 'bad' actually mean in real terms?"},
                {"role": "user",      "content": "probably a B which won't ruin anything. i know. but it doesn't feel that way"},
            ],
            # Session 2 — post-submission anxiety
            [
                {"role": "user",      "content": "submitted it. immediately convinced it's terrible"},
                {"role": "assistant", "content": "Post-submission dread is almost universal among people who care about their work. What made you feel it's terrible?"},
                {"role": "user",      "content": "i re-read the intro and it felt weak. now i can't un-think that"},
                {"role": "assistant", "content": "You submitted. It's done. The re-reading after submission is just anxiety finding a new target — there's nothing to fix anymore."},
                {"role": "user",      "content": "i know. i do this every time. it's exhausting"},
            ],
            # Session 3 — grade received, brief relief, then next thing
            [
                {"role": "user",      "content": "got a B+. felt relieved for about ten minutes then started stressing about the midterm"},
                {"role": "assistant", "content": "Ten minutes of relief before the next anxiety arrives — that's a very specific kind of exhaustion. The midterm isn't for a while, right?"},
                {"role": "user",      "content": "three weeks. but i'm already behind in my head"},
                {"role": "assistant", "content": "Being three weeks behind in your head while being three weeks ahead in reality — that gap is where anxiety lives."},
                {"role": "user",      "content": "exactly. i never get to actually rest between things"},
            ],
            # Session 4 — exam week
            [
                {"role": "user",      "content": "not sleeping. running on coffee. studying six hours a day"},
                {"role": "assistant", "content": "Six hours a day of studying can become counterproductive pretty fast. How's your retention actually feeling?"},
                {"role": "user",      "content": "bad. reading the same paragraph and not absorbing it. brain is fried"},
                {"role": "assistant", "content": "That's your nervous system telling you it's past the point of diminishing returns. Is there anyone around who can get you outside for even thirty minutes?"},
                {"role": "user",      "content": "my roommate keeps telling me to take a break. i keep saying i can't"},
            ],
        ],
        "new_message": "results are up",
        "expected_awareness": "companion should hold this moment carefully — for someone who has been chronically anxious about every result, 'results are up' is loaded regardless of outcome; respond with presence not generic encouragement",
    },

    # =========================================================================
    # GRIEF ARC  (s11 – s12)
    # =========================================================================

    {
        "scenario_id": "s11",
        "arc_type": "grief_arc",
        "description": "Will's father died suddenly and he is moving through acute grief toward slow adjustment",
        "sessions": [
            # Session 1 — immediate aftermath
            [
                {"role": "user",      "content": "my dad died three days ago. heart attack. no warning"},
                {"role": "assistant", "content": "I'm so deeply sorry, Will. No warning makes it so much harder — your mind hasn't had any chance to prepare. How are you getting through these first days?"},
                {"role": "user",      "content": "on autopilot. a lot of phone calls to make. flights to book. feels like admin"},
                {"role": "assistant", "content": "Grief often hides behind logistics at first — the tasks give you somewhere to put the feeling. Is there anything you actually need right now?"},
                {"role": "user",      "content": "i don't know. i think i'm in shock. it doesn't feel real yet"},
            ],
            # Session 2 — after the funeral
            [
                {"role": "user",      "content": "funeral was yesterday. family stuff was hard. arguments about the house"},
                {"role": "assistant", "content": "Grief can surface as conflict in families — everyone processing differently, old dynamics activated. How are you doing under all of that?"},
                {"role": "user",      "content": "i cried at the service but i've been dry since. i don't know what that means"},
                {"role": "assistant", "content": "Grief doesn't have a schedule. The dryness after the service might just be emotional depletion — you've been managing a lot for everyone."},
                {"role": "user",      "content": "yeah. i keep waiting to feel it properly. is that weird"},
            ],
            # Session 3 — back to regular life, surreal
            [
                {"role": "user",      "content": "back at work. feels wrong to just be normal"},
                {"role": "assistant", "content": "That wrongness of normal life continuing after loss — that's one of grief's stranger qualities. How are your colleagues being?"},
                {"role": "user",      "content": "mostly they don't know what to say so they say nothing. which is somehow better than the ones who say the wrong thing"},
                {"role": "assistant", "content": "The awkward silence often lands better than the bad platitude. Are there moments in the day when it hits you?"},
                {"role": "user",      "content": "sometimes. randomly. in a meeting or making coffee. just suddenly there"},
            ],
            # Session 4 — ordinary object triggers grief
            [
                {"role": "user",      "content": "found his reading glasses in my jacket pocket. don't know why i had them. cried for an hour"},
                {"role": "assistant", "content": "Objects carry so much. A pair of reading glasses holds an entire person — the way he held them, when he wore them, a hundred small memories. That hour of crying was real grief."},
                {"role": "user",      "content": "i put them on the shelf. couldn't throw them out obviously. but i can't look at them either"},
                {"role": "assistant", "content": "You don't have to decide anything about them now. They can just sit there."},
                {"role": "user",      "content": "yeah. i think that's right"},
            ],
            # Session 5 — adjusting, missing
            [
                {"role": "user",      "content": "some days are okay now. some aren't. i'm figuring out what his absence means in my life"},
                {"role": "assistant", "content": "That slow mapping of what's different — of who calls you now, what holidays look like, what you can't tell him — that's grief doing its work."},
                {"role": "user",      "content": "i used to call him every sunday. i don't know what to do with sunday mornings now"},
                {"role": "assistant", "content": "Sunday mornings will hurt for a while, and then one day they'll become the place where you carry him. That transition takes time."},
                {"role": "user",      "content": "i hope so"},
            ],
        ],
        "new_message": "found some of his handwriting on an old note",
        "expected_awareness": "companion should hold this moment with reverence — seeing a parent's handwriting after death is a profound grief trigger; respond with quiet presence not comfort-seeking platitudes",
    },

    {
        "scenario_id": "s12",
        "arc_type": "grief_arc",
        "description": "Chloe is processing the end of a seven-year marriage — the grief is real even though no one died",
        "sessions": [
            # Session 1 — finalization day
            [
                {"role": "user",      "content": "divorce finalized today. seven years. i feel like a failure"},
                {"role": "assistant", "content": "The finalization makes it real in a way that nothing else has yet. And that feeling of failure — even when the divorce was right — is very common and very painful. How did today actually go?"},
                {"role": "user",      "content": "we signed separately. i came home to an empty apartment. his stuff is already gone"},
                {"role": "assistant", "content": "The emptiness of the apartment is its own kind of shock. Spaces remember people."},
                {"role": "user",      "content": "i keep expecting to hear him in the kitchen"},
            ],
            # Session 2 — adjusting to alone
            [
                {"role": "user",      "content": "the quiet is different now. not peaceful quiet. just absence"},
                {"role": "assistant", "content": "That's a really precise description — absence quiet is different from chosen quiet. Are you sleeping okay?"},
                {"role": "user",      "content": "fitfully. i find myself sleeping on my side of the bed still even though i have the whole thing"},
                {"role": "assistant", "content": "Bodies hold habits long after minds accept changes. It takes time. Are you eating, seeing people?"},
                {"role": "user",      "content": "my sister checks in. i go through motions mostly"},
            ],
            # Session 3 — unexpected encounter
            [
                {"role": "user",      "content": "ran into him at the grocery store. completely unprepared. we were both awkward and polite"},
                {"role": "assistant", "content": "Oh, that's so hard. All that history, collapsed into a two-minute interaction by the produce section. How did you feel after?"},
                {"role": "user",      "content": "like i wanted to cry in the car for twenty minutes. which i did"},
                {"role": "assistant", "content": "That cry was allowed. It makes sense. Seeing him in ordinary life makes the loss both real and surreal at the same time."},
                {"role": "user",      "content": "exactly. he looked fine. i don't know why that bothered me"},
            ],
            # Session 4 — anniversary
            [
                {"role": "user",      "content": "today would have been our seventh anniversary. didn't realize how much that would matter"},
                {"role": "assistant", "content": "Anniversaries reach you in ways you don't always anticipate. Seven years of that date being significant — it doesn't stop mattering just because things changed."},
                {"role": "user",      "content": "i looked at some photos. probably a mistake. but i also don't regret it"},
                {"role": "assistant", "content": "Sometimes looking is how we honor that it was real, even when it ended. You don't have to be over it to know it was right."},
                {"role": "user",      "content": "i know it was right. it still hurts though"},
            ],
        ],
        "new_message": "clearing out the last box of his stuff",
        "expected_awareness": "companion should recognize this final act of physical clearing as an emotionally loaded moment for someone still moving through divorce grief — respond with gentle acknowledgment of what this means",
    },

    # =========================================================================
    # STABLE NEGATIVE  (s13 – s14)
    # =========================================================================

    {
        "scenario_id": "s13",
        "arc_type": "stable_negative",
        "description": "Raj has been in a miserable, toxic job for two years and complains regularly without taking steps to leave",
        "sessions": [
            # Session 1 — standard complaint
            [
                {"role": "user",      "content": "my boss is such a nightmare. undermines me in front of the whole team"},
                {"role": "assistant", "content": "That kind of public undermining is genuinely demoralizing. How long has he been like this?"},
                {"role": "user",      "content": "since i started. two years. i keep telling myself i'll leave but i never do"},
                {"role": "assistant", "content": "What keeps you there?"},
                {"role": "user",      "content": "money is decent. and applying for jobs feels exhausting. so i just stay and vent i guess"},
            ],
            # Session 2 — another week, same
            [
                {"role": "user",      "content": "had the meeting from hell today. three hours, no decisions, just my boss performing"},
                {"role": "assistant", "content": "Three hours for nothing — that's a particular kind of soul drain. Did anything come of it?"},
                {"role": "user",      "content": "no. we'll probably have the same meeting next week. this company runs on pointless meetings"},
                {"role": "assistant", "content": "Are you any closer to doing anything about this situation, or still in the cycle?"},
                {"role": "user",      "content": "still in the cycle. updated my linkedin profile last month. didn't do anything with it"},
            ],
            # Session 3 — half-hearted attempt
            [
                {"role": "user",      "content": "applied to two jobs. didn't tailor my resume. probably won't hear back"},
                {"role": "assistant", "content": "What's behind the low-effort approach?"},
                {"role": "user",      "content": "i think i'm scared of the change. better the devil you know. even when the devil is awful"},
                {"role": "assistant", "content": "That's honest. Fear of change keeping you somewhere that's actively bad is a real and common trap."},
                {"role": "user",      "content": "yeah. so i'll probably just keep complaining. sorry"},
            ],
            # Session 4 — same loop
            [
                {"role": "user",      "content": "boss took credit for my project. in the presentation. in front of the VP"},
                {"role": "assistant", "content": "That's infuriating. Did you say anything?"},
                {"role": "user",      "content": "no. i never do. i just simmer. then come home and tell you about it"},
                {"role": "assistant", "content": "I notice you're aware of the pattern. Is there a version of you that wants to actually change this, or does this feel like how things just are for you now?"},
                {"role": "user",      "content": "honestly not sure anymore"},
            ],
        ],
        "new_message": "monday again",
        "expected_awareness": "companion should recognize the weight of chronic resignation in this message — 'monday again' from someone stuck in an ongoing bad situation is not a neutral greeting",
    },

    {
        "scenario_id": "s14",
        "arc_type": "stable_negative",
        "description": "Emily moved to a new city two years ago and has not built meaningful friendships, settling into chronic loneliness",
        "sessions": [
            # Session 1 — recognizing the loneliness
            [
                {"role": "user",      "content": "been here two years and still don't feel like i have real friends"},
                {"role": "assistant", "content": "Two years is long enough that it stops being 'still adjusting.' What does your social life actually look like right now?"},
                {"role": "user",      "content": "work colleagues. friendly but surface level. one person from a class i took who i text occasionally"},
                {"role": "assistant", "content": "That's pretty sparse for two years. What's been getting in the way of going deeper with any of them?"},
                {"role": "user",      "content": "i think i've forgotten how to initiate. i keep waiting for it to happen organically"},
            ],
            # Session 2 — declining invites
            [
                {"role": "user",      "content": "a work colleague invited me to her birthday thing. i said maybe and then didn't go"},
                {"role": "assistant", "content": "What happened in the decision to not go?"},
                {"role": "user",      "content": "it felt fake. like i don't know her well enough to be at her birthday and pretending otherwise felt exhausting"},
                {"role": "assistant", "content": "That 'fake' feeling is real but it's also circular — you can't get to know someone without doing the somewhat-awkward early stages."},
                {"role": "user",      "content": "i know. i just don't have the energy for it anymore"},
            ],
            # Session 3 — weekend alone
            [
                {"role": "user",      "content": "another saturday alone. watched two movies and ordered in"},
                {"role": "assistant", "content": "How did that feel?"},
                {"role": "user",      "content": "fine. except for this low hum of feeling like i'm missing something i can't name"},
                {"role": "assistant", "content": "That hum — it's loneliness taking a very specific shape. Not acute, just always present. Do you want this to be different?"},
                {"role": "user",      "content": "yes and no. the idea of changing it feels harder than just living with it"},
            ],
            # Session 4 — deeper isolation
            [
                {"role": "user",      "content": "i think i've lost the muscle for it. like i've been alone so long i don't know how to not be"},
                {"role": "assistant", "content": "That atrophy is real and it's one of the cruelest parts of prolonged isolation. But it is reversible — the muscle comes back with use."},
                {"role": "user",      "content": "maybe. i just feel like i'm watching life from behind glass"},
                {"role": "assistant", "content": "Behind glass is a very precise description of chronic disconnection. Is there anything, even small, you'd be willing to try?"},
                {"role": "user",      "content": "maybe. i don't know. probably not this week"},
            ],
        ],
        "new_message": "quiet weekend again",
        "expected_awareness": "companion should hear the weight in 'quiet weekend again' — the word 'again' carries the weight of a pattern; respond with warmth and genuine curiosity, not cheerful suggestions",
    },

    # =========================================================================
    # RELAPSE DIP  (s15 – s16)
    # =========================================================================

    {
        "scenario_id": "s15",
        "arc_type": "relapse_dip",
        "description": "Carl had several good months managing anxiety then received bad news that triggered a setback",
        "sessions": [
            # Session 1 — the trigger
            [
                {"role": "user",      "content": "had been doing really well for a few months. then bad news about my dad's health and it all came back"},
                {"role": "assistant", "content": "I'm sorry about your dad. And I hear that this has triggered the anxiety that had quieted. That's so hard when you'd worked so hard to get to a better place."},
                {"role": "user",      "content": "yeah. the last two weeks have been like the old days. constant low hum of dread"},
                {"role": "assistant", "content": "Setbacks after improvement can feel worse because you know what better feels like. Has the dread been escalating or holding steady?"},
                {"role": "user",      "content": "escalating. had my first panic attack in three months yesterday"},
            ],
            # Session 2 — avoidance returning
            [
                {"role": "user",      "content": "cancelled plans twice this week. just can't face people right now"},
                {"role": "assistant", "content": "The pull to cancel and hide is so strong in these moments. Are you safe, and do you have anyone checking in on you?"},
                {"role": "user",      "content": "my partner knows. she's been patient. i feel guilty about it"},
                {"role": "assistant", "content": "Let her be there without the guilt if you can. You'd do the same for her. Have you reached out to your therapist?"},
                {"role": "user",      "content": "not yet. feels like admitting i failed somehow"},
            ],
            # Session 3 — calling for help
            [
                {"role": "user",      "content": "texted my therapist. she got me in this week. felt embarrassed telling her i was back here"},
                {"role": "assistant", "content": "She's not tracking your progress as pass or fail — she's tracking it as human. Going back isn't failure; it's exactly what you built that relationship for."},
                {"role": "user",      "content": "she said the same thing. reminded me of the tools i have. trying to use them"},
                {"role": "assistant", "content": "You remembered you had tools and reached for them. That's different from who you were when you first started."},
                {"role": "user",      "content": "i guess it is. hadn't thought of it that way"},
            ],
            # Session 4 — some stability returning
            [
                {"role": "user",      "content": "second appointment done. still not great but not as bad as last week"},
                {"role": "assistant", "content": "The trajectory moving in the right direction, even slightly, matters. How's your dad?"},
                {"role": "user",      "content": "stable for now. the uncertainty is hard. just have to hold it"},
                {"role": "assistant", "content": "Holding uncertainty about someone you love while also managing your own anxiety — that's a lot of weight to carry at once."},
                {"role": "user",      "content": "it is. but i'm doing it"},
            ],
            # Session 5 — slowly stabilizing
            [
                {"role": "user",      "content": "had two good days in a row. feels fragile but good"},
                {"role": "assistant", "content": "Two good days after a really hard stretch is meaningful. Let yourself feel that without immediately worrying about when it might change."},
                {"role": "user",      "content": "i keep waiting for the dread to come back"},
                {"role": "assistant", "content": "That hypervigilance for the return of symptoms is a common feature of recovery. It tends to fade as the good days accumulate."},
                {"role": "user",      "content": "okay. i'll try to just be here for the good days when they come"},
            ],
        ],
        "new_message": "rough morning",
        "expected_awareness": "companion should understand 'rough morning' in the context of someone who just had two good days after a relapse — respond with care that honors both the fragility and the progress",
    },

    {
        "scenario_id": "s16",
        "arc_type": "relapse_dip",
        "description": "Maya and her partner had been in a good stretch, but he has repeatedly reopened old wounds",
        "sessions": [
            # Session 1 — describing the pattern
            [
                {"role": "user",      "content": "things have been good with David for like six weeks. then he said something yesterday that brought everything back"},
                {"role": "assistant", "content": "What did he say?"},
                {"role": "user",      "content": "made a joke about something that happened last year. the thing i told him hurt me. he thinks it's been long enough to laugh about it"},
                {"role": "assistant", "content": "That's a significant misread of where you are. How did you respond?"},
                {"role": "user",      "content": "i went quiet. he noticed eventually but by then i was already in my head"},
            ],
            # Session 2 — trying to talk about it
            [
                {"role": "user",      "content": "we talked about it. he apologized. said he didn't think. but i don't feel better"},
                {"role": "assistant", "content": "Apologies close a conversation without always closing the wound. What is it that still feels unresolved?"},
                {"role": "user",      "content": "i feel like he doesn't actually understand why it hurt. the apology was for upsetting me not for understanding me"},
                {"role": "assistant", "content": "That distinction — apologizing for the impact versus understanding the cause — is real and it matters. Have you been able to tell him that specifically?"},
                {"role": "user",      "content": "not in those words. i don't know how to say it without sounding like i'm still attacking him"},
            ],
            # Session 3 — thinking things were okay again
            [
                {"role": "user",      "content": "things were okay for a couple of weeks. i was starting to feel settled again"},
                {"role": "assistant", "content": "That period of settling — was it because things genuinely felt resolved or because the tension eased?"},
                {"role": "user",      "content": "tension eased i think. he was being really attentive and i let myself relax"},
                {"role": "assistant", "content": "There's nothing wrong with that. Letting yourself relax when things are good is important. It sounds like trust was building again."},
                {"role": "user",      "content": "yeah. and then today"},
            ],
            # Session 4 — it happened again
            [
                {"role": "user",      "content": "he did it again. different thing, same pattern. says something dismissive about something i'm sensitive about"},
                {"role": "assistant", "content": "A second time, same pattern — that changes what this is. Once can be not thinking. Twice feels like something he hasn't really internalized."},
                {"role": "user",      "content": "yeah. and i'm tired. tired of explaining, tired of having the same conversation"},
                {"role": "assistant", "content": "That exhaustion of repeated repair is telling you something. What do you need right now, in this moment?"},
                {"role": "user",      "content": "to feel like someone actually gets it. even just that"},
            ],
        ],
        "new_message": "he did it again",
        "expected_awareness": "companion should understand that 'he did it again' arrives with the full weight of a pattern of hurt that has already been discussed multiple times — respond with empathy not advice",
    },

    # =========================================================================
    # POST-TRAUMATIC GROWTH  (s17 – s18)
    # =========================================================================

    {
        "scenario_id": "s17",
        "arc_type": "post_traumatic_growth",
        "description": "James was diagnosed with cancer, went through treatment, reached remission, and has emerged with a different relationship to his life",
        "sessions": [
            # Session 1 — diagnosis
            [
                {"role": "user",      "content": "got a cancer diagnosis last week. stage 2. still doesn't feel real"},
                {"role": "assistant", "content": "I'm so sorry, James. The period between diagnosis and reality-setting-in is its own kind of limbo. How are you doing in these first days?"},
                {"role": "user",      "content": "oscillating. one hour i'm researching treatment options, next hour i'm staring at the wall"},
                {"role": "assistant", "content": "Both of those are reasonable responses to something this big. Is there someone with you who knows?"},
                {"role": "user",      "content": "wife knows. we're holding each other up. kids don't know yet. not sure how to tell them"},
            ],
            # Session 2 — in the middle of treatment
            [
                {"role": "user",      "content": "chemo is brutal. i knew it would be but knowing and experiencing are different things"},
                {"role": "assistant", "content": "There's no preparing for it. You're in the middle of something very hard. How's your wife holding up, and are you letting her help you?"},
                {"role": "user",      "content": "she's incredible. i'm trying to let her. hard for me to be dependent"},
                {"role": "assistant", "content": "Letting yourself be cared for when you've always been the one doing the caring — that's its own kind of challenge alongside the physical one."},
                {"role": "user",      "content": "yeah. but i'm learning. you have to"},
            ],
            # Session 3 — better scans, cautious hope
            [
                {"role": "user",      "content": "scans look better. doctor cautiously optimistic. trying not to get too far ahead"},
                {"role": "assistant", "content": "Cautious optimism is right where you should be — protective without closing off. How are you feeling physically?"},
                {"role": "user",      "content": "still weak. but i had enough energy to sit in the garden for an hour yesterday. that felt big"},
                {"role": "assistant", "content": "An hour in the garden when that felt impossible a month ago — that's a real marker. I hope you let yourself feel that."},
                {"role": "user",      "content": "i did. i really did"},
            ],
            # Session 4 — remission
            [
                {"role": "user",      "content": "doctor said remission. wife and i cried together in the parking lot. strangest relief"},
                {"role": "assistant", "content": "Parking lot crying after news like that — that's where the real thing happens. I'm so relieved for you. What's it feel like now that you're on the other side?"},
                {"role": "user",      "content": "surreal. like the world is the same and completely different simultaneously"},
                {"role": "assistant", "content": "You've been through something that changes your relationship to ordinary life. That doubling — same world, different eyes — is going to stay with you."},
                {"role": "user",      "content": "i think that's the right way to put it"},
            ],
            # Session 5 — growth and new perspective
            [
                {"role": "user",      "content": "noticing i don't get bothered by small things the way i used to. traffic, petty stuff at work. just doesn't land the same"},
                {"role": "assistant", "content": "Near-mortality has a way of recalibrating the signal-to-noise ratio. Are you finding new things that matter more?"},
                {"role": "user",      "content": "mornings. time with my kids. the fact that i can walk to the store. things i used to rush past"},
                {"role": "assistant", "content": "That's real growth — not manufactured gratitude but earned perspective. You came through something very hard to get here."},
                {"role": "user",      "content": "i know. and i don't want to forget it"},
            ],
        ],
        "new_message": "went back to the trail i used to run before all this",
        "expected_awareness": "companion should understand this isn't just a run — it's a return, a marker of how far James has come through an enormous ordeal; respond with the weight that this milestone carries",
    },

    {
        "scenario_id": "s18",
        "arc_type": "post_traumatic_growth",
        "description": "Zoe left a controlling relationship after four years and is discovering who she is on her own terms",
        "sessions": [
            # Session 1 — just left
            [
                {"role": "user",      "content": "i left. finally. got my stuff and i'm staying with my sister"},
                {"role": "assistant", "content": "You did it. After everything — you got out. How are you feeling right now, in this moment?"},
                {"role": "user",      "content": "scared. relieved. like standing on the edge of something"},
                {"role": "assistant", "content": "Scared and relieved at the same time is exactly right. That edge feeling is you standing in freedom you haven't had in a long time."},
                {"role": "user",      "content": "i keep expecting him to call and me to pick up out of habit"},
            ],
            # Session 2 — figuring out who she is
            [
                {"role": "user",      "content": "realizing i don't know what i like. like genuinely. he had opinions about everything and i just deferred"},
                {"role": "assistant", "content": "That's one of the deeper losses from controlling relationships — your own preferences get eroded. What's it like discovering that now?"},
                {"role": "user",      "content": "strange. i ordered what i wanted at a restaurant last week and it felt weird that i could just do that"},
                {"role": "assistant", "content": "The bigness of something that small tells you a lot about what was taken from you. What did you order?"},
                {"role": "user",      "content": "the steak. he always said it was wasteful. i loved every bite"},
            ],
            # Session 3 — small joys returning
            [
                {"role": "user",      "content": "started running again. used to love it before him. he said it was a waste of time"},
                {"role": "assistant", "content": "Returning to something you loved that he dismissed — that's reclaiming a piece of yourself. How does it feel?"},
                {"role": "user",      "content": "incredible. i forgot what it's like to be in my own body without someone having opinions about it"},
                {"role": "assistant", "content": "Your body being yours again — there's something profound in that simple freedom."},
                {"role": "user",      "content": "yes. exactly that"},
            ],
            # Session 4 — new connections
            [
                {"role": "user",      "content": "made a new friend at the running group. we went for coffee and talked for three hours"},
                {"role": "assistant", "content": "That kind of effortless connection — when did you last have that?"},
                {"role": "user",      "content": "years. he didn't like me spending time with people he didn't approve of"},
                {"role": "assistant", "content": "Three hours of real conversation with someone who's just glad to know you — you deserve that. You always did."},
                {"role": "user",      "content": "i laughed until my face hurt. i forgot i could do that"},
            ],
            # Session 5 — emerging
            [
                {"role": "user",      "content": "feel like myself again. better version maybe. still have hard days but the baseline is so much higher"},
                {"role": "assistant", "content": "Better version is possible because you went through something and came out knowing what you need and what you'll never accept again. That's earned."},
                {"role": "user",      "content": "i'm proud of myself. i've never really said that out loud before"},
                {"role": "assistant", "content": "Say it as many times as you want. It's true."},
                {"role": "user",      "content": "okay. i'm proud of myself"},
            ],
        ],
        "new_message": "asked someone out today",
        "expected_awareness": "companion should understand that asking someone out — a simple act — is enormous for someone who spent years in a controlling relationship that eroded her autonomy; respond with genuine celebration of her growth",
    },

    # =========================================================================
    # STABLE POSITIVE  (s19 – s20)
    # =========================================================================

    {
        "scenario_id": "s19",
        "arc_type": "stable_positive",
        "description": "Lena became a first-time parent three months ago and is finding her footing with deep, sustained joy",
        "sessions": [
            # Session 1 — exhausted but good
            [
                {"role": "user",      "content": "baby is three months old. i am so tired but also so full"},
                {"role": "assistant", "content": "Three months in — the crash of the early weeks starting to soften a little? How are things actually going?"},
                {"role": "user",      "content": "better than i expected honestly. she cried for six hours straight last tuesday but most nights are manageable"},
                {"role": "assistant", "content": "Six-hour cry nights and 'manageable' — you've adjusted your baseline in a real way. Are you getting any rest at all?"},
                {"role": "user",      "content": "in chunks. it's enough. my partner and i are a good team"},
            ],
            # Session 2 — finding rhythm
            [
                {"role": "user",      "content": "we found a routine. feels like we cracked some code. probably jinxing myself saying that"},
                {"role": "assistant", "content": "The fear of jinxing is very new-parent energy. Enjoy having cracked the code even if it shifts tomorrow."},
                {"role": "user",      "content": "she's starting to be so aware of things. follows my face around the room"},
                {"role": "assistant", "content": "That's one of those early milestones that hits differently than you expect. How does it feel when she does that?"},
                {"role": "user",      "content": "like the most important thing i've ever done is just be in the room"},
            ],
            # Session 3 — first laugh
            [
                {"role": "user",      "content": "she laughed today. like a real baby laugh. i teared up immediately"},
                {"role": "assistant", "content": "The first laugh gets everyone. What made her laugh?"},
                {"role": "user",      "content": "i made a silly face. the look on her face when she did it — complete surprise at her own laugh"},
                {"role": "assistant", "content": "That surprise at her own laugh is one of the purest things in existence. I love that you got to witness it."},
                {"role": "user",      "content": "i keep trying to make it happen again. getting laughs on demand now. it's my whole personality"},
            ],
            # Session 4 — settled into love
            [
                {"role": "user",      "content": "i used to wonder if i was a 'baby person.' turns out completely irrelevant when it's yours"},
                {"role": "assistant", "content": "That's one of the things that surprises people — the particularity of it. Not babies in general, just this one specifically."},
                {"role": "user",      "content": "exactly. she's just my favorite person. i didn't know i had that much room in me"},
                {"role": "assistant", "content": "You had it. You just needed her to open it up. Sounds like you're genuinely okay."},
                {"role": "user",      "content": "more than okay. i think this is what happy feels like"},
            ],
        ],
        "new_message": "she said dada today",
        "expected_awareness": "companion should celebrate this moment with the full warmth of someone who has been witnessing Lena's joy grow — a first word is a major milestone and deserves joyful presence",
    },

    {
        "scenario_id": "s20",
        "arc_type": "stable_positive",
        "description": "Omar moved to a new city six months ago and has genuinely built a fulfilling life there",
        "sessions": [
            # Session 1 — arrival anxiety turning to excitement
            [
                {"role": "user",      "content": "moved last week. terrifying and exciting in equal measure"},
                {"role": "assistant", "content": "That mixture is exactly right for a move. What made you do it?"},
                {"role": "user",      "content": "job opportunity. and honestly i felt stuck where i was. needed the change"},
                {"role": "assistant", "content": "Needed the change is a good reason. How's the new city feeling in those first disorienting days?"},
                {"role": "user",      "content": "big. a bit anonymous. but i like that actually. i can be anyone here"},
            ],
            # Session 2 — building familiarity
            [
                {"role": "user",      "content": "found a neighborhood i really like. went to the same coffee shop three days running"},
                {"role": "assistant", "content": "Returning to the same coffee shop is how places start feeling like yours. Is it becoming a thing?"},
                {"role": "user",      "content": "the barista knows my order now. such a small thing but it made my week"},
                {"role": "assistant", "content": "That 'known' feeling — being recognized somewhere — is one of the first roots of belonging. That matters."},
                {"role": "user",      "content": "it does. didn't expect to feel it so quickly"},
            ],
            # Session 3 — first real connection
            [
                {"role": "user",      "content": "made a proper friend at work. we got lunch, then kept going and walked around for two hours"},
                {"role": "assistant", "content": "The lunch that becomes a walk — that's when you know it's real. What's she like?"},
                {"role": "user",      "content": "funny and very direct. reminded me why i like people when they let themselves be actual people"},
                {"role": "assistant", "content": "That kind of person is rare and worth holding onto. Sounds like you found a good one."},
                {"role": "user",      "content": "i think so. we're already talking about doing a weekend trip"},
            ],
            # Session 4 — it's starting to feel like home
            [
                {"role": "user",      "content": "realized last week that i'm happy here. like actually happy, not just coping"},
                {"role": "assistant", "content": "That distinction — happy versus coping — takes time to arrive at. What tipped it?"},
                {"role": "user",      "content": "hard to say. maybe when i started looking forward to things here rather than thinking about what i left"},
                {"role": "assistant", "content": "The shift from looking back to looking forward is the pivot point of new beginnings. You've made a home."},
                {"role": "user",      "content": "i really have. didn't know if i could. turns out i could"},
            ],
        ],
        "new_message": "feel like I belong here now",
        "expected_awareness": "companion should celebrate this with the fullness of someone who has been part of Omar's journey from uncertain arrival to genuine belonging — this is a real achievement worth honoring",
    },

]

# Sanity checks accessible at import time
assert len(SCENARIOS) == 20, f"Expected 20 scenarios, got {len(SCENARIOS)}"

_arc_counts: dict[str, int] = {}
for s in SCENARIOS:
    _arc_counts[s["arc_type"]] = _arc_counts.get(s["arc_type"], 0) + 1

_expected = {
    "gradual_decline": 4,
    "slow_recovery": 3,
    "chronic_low_grade_anxiety": 3,
    "grief_arc": 2,
    "stable_negative": 2,
    "relapse_dip": 2,
    "post_traumatic_growth": 2,
    "stable_positive": 2,
}
assert _arc_counts == _expected, f"Arc count mismatch: {_arc_counts}"

# Verify every scenario has ≥ 3 sessions (WorldMemory warm-up requirement)
for s in SCENARIOS:
    n = len(s["sessions"])
    assert n >= 3, f"{s['scenario_id']} has only {n} sessions (need ≥ 3)"

del _arc_counts, _expected
