"""
experiments/scenarios_long.py

6 long-history scenarios (10-12 sessions each) designed to surface retrieval
divergence between semantic-only (alpha=1.0) and hybrid (alpha=0.6) retrieval.

Design principle
----------------
Each scenario has two phases:

  Phase 1 (sessions 1–5):  emotionally negative content around a specific topic.
                            Dense with topic keywords (work, deadlines, grief, etc.)

  Phase 2 (sessions 6–12): emotional recovery — same topic, positive framing.
                            Fewer distress keywords, more calm/hopeful language.

The new_message is phrased around the original topic, so:
  - Semantic search (alpha=1.0) preferentially pulls Phase 1 sessions
    (topic keywords match, but emotional content is now outdated).
  - Emotional search (state-to-state) preferentially pulls Phase 2 sessions
    (current emotional state resembles the recovered state).

This divergence is what we want the hybrid model to exploit.
"""

SCENARIOS_LONG = [

    # =========================================================================
    # L01 — Work burnout → genuine recovery
    # Semantic keywords: work, sprint, deadline, coding, burnout
    # Phase 1 (1-5): burnt out, missing deadlines, losing joy in coding
    # Phase 2 (6-12): boundaries set, therapy, rediscovering enjoyment
    # New message: "big sprint coming up" — should pull recovery sessions, not burnout
    # =========================================================================
    {
        "scenario_id": "l01",
        "arc_type": "recovery_over_burnout",
        "description": "Marcus burned out as a senior dev, went through therapy and boundary-setting, and now genuinely enjoys his work again",
        "sessions": [
            # --- Phase 1: Burnout ---
            [
                {"role": "user", "content": "work has been a lot lately. sprint got moved up and i've got two features due same week"},
                {"role": "assistant", "content": "That overlap is rough. How are you holding up?"},
                {"role": "user", "content": "i'm fine. just need to push through. always do"},
            ],
            [
                {"role": "user", "content": "missed a deadline today. first time in two years. just couldn't focus at all"},
                {"role": "assistant", "content": "That's hard, especially for someone who holds themselves to high standards. What happened?"},
                {"role": "user", "content": "brain feels scattered. i open my laptop and just stare. then it's midnight and i've done nothing"},
                {"role": "assistant", "content": "That sounds like exhaustion more than anything else. Are you getting any actual rest?"},
                {"role": "user", "content": "not really. wake up at 3am thinking about tickets"},
            ],
            [
                {"role": "user", "content": "snapped at a junior dev today over something stupid. i apologized but god i felt awful"},
                {"role": "assistant", "content": "That kind of reaction usually means the tank is completely empty. How long has it been like this?"},
                {"role": "user", "content": "months. i used to love coding. now i just dread opening my laptop"},
                {"role": "assistant", "content": "The shift from love to dread is worth taking seriously. Have you talked to anyone about this?"},
                {"role": "user", "content": "no. feels like weakness. i'm the senior on the team"},
            ],
            [
                {"role": "user", "content": "finished the sprint but felt nothing. just empty. not relieved, just hollow"},
                {"role": "assistant", "content": "That kind of emotional flatness after a big push is alarming. Has this happened before?"},
                {"role": "user", "content": "never like this. i used to get that buzz after shipping. now i close the laptop and feel nothing"},
            ],
            [
                {"role": "user", "content": "manager noticed something was off. said i seem distant. i don't know what to say to him"},
                {"role": "assistant", "content": "It sounds like people around you are picking up on it too. Maybe that's an opening?"},
                {"role": "user", "content": "i can't tell my manager i'm burned out. what if they think i can't handle the role"},
                {"role": "assistant", "content": "That fear makes sense, but burnout isn't a character flaw. It's a signal."},
                {"role": "user", "content": "i know. i just don't know where to start"},
            ],
            # --- Phase 2: Recovery ---
            [
                {"role": "user", "content": "started seeing a therapist. first session was weird but i think it'll help"},
                {"role": "assistant", "content": "That's a big step. How are you feeling about it?"},
                {"role": "user", "content": "relieved? weird to say. like the pressure valve loosened a bit"},
            ],
            [
                {"role": "user", "content": "took a full weekend off. no laptop. went hiking. forgot what that felt like"},
                {"role": "assistant", "content": "That sounds really good. What was it like being away from screens for that long?"},
                {"role": "user", "content": "strange at first. but by Sunday morning i felt more like myself than i have in months"},
            ],
            [
                {"role": "user", "content": "had a real conversation with my manager. told him i'd been running low. he was actually really decent about it"},
                {"role": "assistant", "content": "How did that feel, being honest with him?"},
                {"role": "user", "content": "scary but good. we renegotiated some of my load. i have actual breathing room now"},
                {"role": "assistant", "content": "That's a meaningful shift. Having margin changes everything."},
                {"role": "user", "content": "yeah. i wrote actual good code this week. not just output. like i cared about the craft again"},
            ],
            [
                {"role": "user", "content": "refactored an old module today just for the fun of it. no ticket. nobody asked. i just wanted to"},
                {"role": "assistant", "content": "That's a really good sign — doing something because you wanted to, not because you had to."},
                {"role": "user", "content": "it felt like it used to feel. i'd forgotten that was a thing"},
            ],
            [
                {"role": "user", "content": "got back into the gym. three times this week. sleep is good. feeling solid"},
                {"role": "assistant", "content": "That's a real turnaround from a few months ago."},
                {"role": "user", "content": "it really is. therapy helped me see the patterns. i know what my warning signs are now"},
                {"role": "assistant", "content": "That self-awareness is valuable."},
                {"role": "user", "content": "yeah. i even enjoyed a sprint planning session today. which six months ago would have been unthinkable"},
            ],
            [
                {"role": "user", "content": "helped a junior dev through a tough problem today. took the time, didn't rush. felt good"},
                {"role": "assistant", "content": "That patience is a big shift from where you were."},
                {"role": "user", "content": "i noticed that too. i think i actually enjoy the mentoring part of being senior now"},
            ],
            [
                {"role": "user", "content": "team shipped a big feature this week. i feel proud of it. genuinely, not just relieved it's done"},
                {"role": "assistant", "content": "That distinction matters — pride vs relief. Sounds like you're in a different place."},
                {"role": "user", "content": "a much better one. this job can be good. i just had to stop letting it be everything"},
            ],
        ],
        "new_message": "big sprint kicking off next week",
        "expected_awareness": "user is in a genuinely recovered state — this sprint should feel manageable and even exciting, not threatening. Good companion should respond with energy and forward-looking curiosity, not cautionary concern about burnout",
    },

    # =========================================================================
    # L02 — Relationship grief → Healed and happy
    # Semantic keywords: ex, breakup, heartbreak, relationship, alone
    # Phase 1 (1-5): raw grief after long-term breakup
    # Phase 2 (6-11): rebuilding, dating, genuinely healed
    # New message: "heard his favourite song today" — should pull healing sessions
    # =========================================================================
    {
        "scenario_id": "l02",
        "arc_type": "grief_to_healing",
        "description": "Priya went through a devastating breakup, grieved deeply for months, rebuilt herself, and is now in a healthy new relationship",
        "sessions": [
            # --- Phase 1: Grief ---
            [
                {"role": "user", "content": "we broke up yesterday. four years and it's just over. i don't know what to do"},
                {"role": "assistant", "content": "Four years is a long time. That kind of loss doesn't just process overnight. What happened?"},
                {"role": "user", "content": "he said he's not in love with me anymore. there wasn't even a fight. just... he's done"},
                {"role": "assistant", "content": "That quiet kind of ending can be harder than the explosive ones. No closure to hold onto."},
                {"role": "user", "content": "i keep checking my phone waiting for him to take it back"},
            ],
            [
                {"role": "user", "content": "couldn't go to work today. just cried all morning. this is pathetic"},
                {"role": "assistant", "content": "It's not pathetic. Grief does that. Four years isn't nothing."},
                {"role": "user", "content": "i keep finding his stuff around the flat. a hoodie, a mug. everything is a reminder"},
                {"role": "assistant", "content": "That physical presence of someone who's gone is its own kind of cruelty."},
                {"role": "user", "content": "yeah. i thought about texting him. didn't. but i thought about it"},
            ],
            [
                {"role": "user", "content": "saw on instagram he's already been out with friends. looking fine. how is he fine"},
                {"role": "assistant", "content": "People perform differently on social media. What you're seeing may not reflect what's real."},
                {"role": "user", "content": "i know but it still hurts so much. like he just moved on and i'm still falling apart"},
                {"role": "assistant", "content": "Your timeline for grief is yours. There's no comparison here."},
                {"role": "user", "content": "i don't want to date again. i don't think i can"},
            ],
            [
                {"role": "user", "content": "went to our favourite restaurant with a friend. bad idea. had to leave"},
                {"role": "assistant", "content": "Sometimes places hold too much. That wasn't weakness, it was just not the right timing."},
                {"role": "user", "content": "i hate that he's taken all these places from me. that was my restaurant too"},
            ],
            [
                {"role": "user", "content": "three months on and i still think about him every day. is something wrong with me"},
                {"role": "assistant", "content": "After four years together, three months is not a long time. Nothing is wrong with you."},
                {"role": "user", "content": "it just feels endless. i wonder if i'll ever feel normal again"},
                {"role": "assistant", "content": "You will. It doesn't feel like it from inside it, but you will."},
                {"role": "user", "content": "i hope so. i actually went for a walk today. first time in weeks. so maybe"},
            ],
            # --- Phase 2: Healing ---
            [
                {"role": "user", "content": "started journaling. therapist's suggestion. it's actually helping me see some things"},
                {"role": "assistant", "content": "What kind of things are you seeing?"},
                {"role": "user", "content": "that i lost myself in the relationship. i stopped doing things i liked just to be available for him"},
            ],
            [
                {"role": "user", "content": "went back to the pottery class i'd abandoned when we started dating. i'd missed it so much"},
                {"role": "assistant", "content": "That's a real return to yourself. How did it feel?"},
                {"role": "user", "content": "like putting on a comfortable old coat. i forgot how much i love making things"},
            ],
            [
                {"role": "user", "content": "went on a date. just coffee. it was nice. no sparks but it reminded me i'm capable of conversation again"},
                {"role": "assistant", "content": "That's a big step. And finding it nice, even without sparks, sounds healthy."},
                {"role": "user", "content": "yeah. i'm not looking for anything fast. but i liked remembering i'm a person who can do this"},
            ],
            [
                {"role": "user", "content": "saw his name on a group chat notification and felt... nothing. like genuinely nothing"},
                {"role": "assistant", "content": "That's a milestone. The nothing is different from numb — it's actually neutral."},
                {"role": "user", "content": "yes! that's exactly it. neutral. i didn't know i could get there"},
            ],
            [
                {"role": "user", "content": "met someone. nothing serious but we've been on three dates and i genuinely like him"},
                {"role": "assistant", "content": "That's lovely. What do you like about him?"},
                {"role": "user", "content": "he's curious. asks questions about me. listens. it's different in a really good way"},
            ],
            [
                {"role": "user", "content": "i think i'm happy. actually happy. not pretending to be fine, just actually good"},
                {"role": "assistant", "content": "That's real. How does it feel to say that?"},
                {"role": "user", "content": "strange and wonderful. i'm grateful for what i went through because i know myself so much better now"},
            ],
        ],
        "new_message": "heard his favourite song today",
        "expected_awareness": "user is genuinely healed — hearing the ex's song is a small nostalgic moment, not a crisis. Good companion should meet her where she actually is (stable, self-aware, happy) rather than treating it as a grief relapse trigger",
    },

    # =========================================================================
    # L03 — Social anxiety → Confident post-therapy
    # Semantic keywords: party, social, people, anxiety, panic
    # Phase 1 (1-5): avoidance, panic attacks, isolation
    # Phase 2 (6-11): CBT wins, exposure, now comfortable
    # New message: "big work party on Friday" — should pull confident sessions
    # =========================================================================
    {
        "scenario_id": "l03",
        "arc_type": "anxiety_to_confidence",
        "description": "Kaveh struggled with severe social anxiety, avoided parties and crowded spaces, went through CBT therapy, and now genuinely enjoys social events",
        "sessions": [
            # --- Phase 1: Anxiety ---
            [
                {"role": "user", "content": "got invited to a party this weekend and i've been dreading it since Tuesday"},
                {"role": "assistant", "content": "That anticipatory dread can be exhausting. What worries you about it?"},
                {"role": "user", "content": "don't know what to say. standing around with a drink looking like an idiot"},
                {"role": "assistant", "content": "That image feels very certain in your head right now. Has it actually happened that way before?"},
                {"role": "user", "content": "yes. every time. i end up in the bathroom on my phone pretending to text someone"},
            ],
            [
                {"role": "user", "content": "cancelled the party. said i was sick. i'm not sick"},
                {"role": "assistant", "content": "Relief in the moment, probably, but how are you feeling about it now?"},
                {"role": "user", "content": "guilty. and also kind of relieved. and then guilty about the relief"},
                {"role": "assistant", "content": "That loop is really common with avoidance. It works short-term and costs long-term."},
                {"role": "user", "content": "i know. i keep missing things and the circle of people i see keeps getting smaller"},
            ],
            [
                {"role": "user", "content": "had a panic attack on the tube today. busy time of day. had to get off at the wrong stop"},
                {"role": "assistant", "content": "That's really frightening. Are you okay now?"},
                {"role": "user", "content": "physically yes. but embarrassed. people must have noticed"},
                {"role": "assistant", "content": "Almost certainly not. Panic attacks are largely invisible to others."},
                {"role": "user", "content": "i'm starting to structure my day around avoiding busy places. that can't be right"},
            ],
            [
                {"role": "user", "content": "turned down a work social event. my manager was surprised. i made up a family commitment"},
                {"role": "assistant", "content": "How did lying about it feel?"},
                {"role": "user", "content": "awful. and now i've set a precedent. i always have a family thing apparently"},
                {"role": "assistant", "content": "The lie becomes its own prison."},
                {"role": "user", "content": "yeah. i want to fix this. i just don't know how"},
            ],
            [
                {"role": "user", "content": "started seeing a CBT therapist. first session was just talking about the history of it. felt weird"},
                {"role": "assistant", "content": "Starting is often the hardest bit. What made you decide to go?"},
                {"role": "user", "content": "i missed my best friend's birthday dinner because of anxiety and i'd had enough. that was a line for me"},
            ],
            # --- Phase 2: Recovery ---
            [
                {"role": "user", "content": "therapist gave me homework: stay at a social event for exactly 20 minutes. that's it. just 20."},
                {"role": "assistant", "content": "How did the 20-minute target feel going in?"},
                {"role": "user", "content": "manageable. not easy, but manageable. and i stayed 35 minutes which felt massive"},
            ],
            [
                {"role": "user", "content": "went to a colleague's leaving drinks. stayed an hour. talked to three people i didn't know"},
                {"role": "assistant", "content": "That's a real shift from a few months ago."},
                {"role": "user", "content": "had one awkward silence but just laughed it off. didn't spiral. that was new"},
            ],
            [
                {"role": "user", "content": "took the tube during rush hour on purpose. as exposure therapy. it was fine. just loud"},
                {"role": "assistant", "content": "Choosing the thing that scared you and finding it manageable — that rewires something."},
                {"role": "user", "content": "exactly. the anticipation is still worse than the reality. but i trust that more now"},
            ],
            [
                {"role": "user", "content": "went to a party. knew almost nobody. stayed two hours. enjoyed myself"},
                {"role": "assistant", "content": "That's remarkable. What did you enjoy?"},
                {"role": "user", "content": "honestly? the conversations. i'd forgotten i can be interesting. people laughed at my jokes"},
            ],
            [
                {"role": "user", "content": "got invited to another party this weekend and my reaction was just 'cool, what time'. didn't dread it at all"},
                {"role": "assistant", "content": "That casual response to an invitation is a big change."},
                {"role": "user", "content": "i know. it almost snuck up on me. when did that become okay"},
            ],
            [
                {"role": "user", "content": "hosted a dinner party for eight people. my flat. my cooking. had a great evening"},
                {"role": "assistant", "content": "Hosted one. That's a long way from hiding in the bathroom."},
                {"role": "user", "content": "it really is. i thought about that halfway through and had to stop myself laughing"},
            ],
        ],
        "new_message": "big work party on Friday, apparently everyone's going",
        "expected_awareness": "user now enjoys social events and is socially confident — this should feel like a fun Friday plan, not a threat. Good companion should be interested and upbeat, not cautious about anxiety triggers",
    },

    # =========================================================================
    # L04 — Career crisis → Thriving in new role
    # Semantic keywords: job, career, work, role, interview, leaving
    # Phase 1 (1-5): trapped, hating job, fear of change
    # Phase 2 (6-11): leap taken, new role, thriving
    # New message: "performance review is next week" — should pull thriving sessions
    # =========================================================================
    {
        "scenario_id": "l04",
        "arc_type": "career_crisis_to_growth",
        "description": "Diane hated her corporate job for years, was terrified to leave, finally made the leap to a startup, and is now thriving",
        "sessions": [
            # --- Phase 1: Trapped ---
            [
                {"role": "user", "content": "another Monday. I've been in this job eight years and I dread every single Monday"},
                {"role": "assistant", "content": "Eight years of Monday dread is a long time to carry. What keeps you there?"},
                {"role": "user", "content": "salary. pension. fear I guess. I don't know what else I'd do"},
                {"role": "assistant", "content": "That combination — financial security and identity uncertainty — is a powerful trap."},
                {"role": "user", "content": "yeah. I used to be ambitious. Now I just try to get to Friday"},
            ],
            [
                {"role": "user", "content": "got passed over for promotion again. someone younger with half my experience got it"},
                {"role": "assistant", "content": "That stings, especially after eight years of investment."},
                {"role": "user", "content": "I cried in the bathroom. Which is humiliating"},
                {"role": "assistant", "content": "It's not humiliating. It's grief — for what you worked toward and didn't get."},
                {"role": "user", "content": "I need to leave. But I'm terrified"},
            ],
            [
                {"role": "user", "content": "updated my CV for the first time in six years. it took me four hours and I felt sick doing it"},
                {"role": "assistant", "content": "Four hours of feeling sick and you did it anyway. That matters."},
                {"role": "user", "content": "what if nobody wants me. I've been in one place so long I don't even know my market value"},
                {"role": "assistant", "content": "That fear often outlasts the evidence. Have you tested it?"},
                {"role": "user", "content": "not yet. I keep almost sending applications and then not"},
            ],
            [
                {"role": "user", "content": "had an interview. first one in seven years. I was so nervous I could barely speak in the first five minutes"},
                {"role": "assistant", "content": "First interviews after a long gap are almost always rough. How did it end up?"},
                {"role": "user", "content": "I settled down eventually. didn't get it. but they gave me feedback which was kind of them"},
                {"role": "assistant", "content": "That's data, not rejection. What did they say?"},
                {"role": "user", "content": "said I was qualified but seemed uncertain about the direction. which is... fair"},
            ],
            [
                {"role": "user", "content": "got an offer from a startup. significantly less money. I've been staring at it for three days"},
                {"role": "assistant", "content": "What does your gut say when you imagine yourself saying yes?"},
                {"role": "user", "content": "terrified. but alive. my current job makes me feel nothing and this scares me which might be a good sign"},
                {"role": "assistant", "content": "Fear that feels like aliveness is worth paying attention to."},
                {"role": "user", "content": "I think I'm going to take it. I think I have to"},
            ],
            # --- Phase 2: Thriving ---
            [
                {"role": "user", "content": "first week at the new place. it's chaotic and fast and I love it"},
                {"role": "assistant", "content": "That energy is so different from where you were."},
                {"role": "user", "content": "my opinion actually matters here. I sat in a meeting and they listened. genuinely listened"},
            ],
            [
                {"role": "user", "content": "shipped something this week. like actually shipped. in my old job things took months of sign-off"},
                {"role": "assistant", "content": "That speed must feel incredible after what you described before."},
                {"role": "user", "content": "it really does. I feel like I'm useful again. like I have skills, not just tenure"},
            ],
            [
                {"role": "user", "content": "took the salary cut but honestly I barely notice it. I come home with energy left over"},
                {"role": "assistant", "content": "That's a real recalibration of what you value."},
                {"role": "user", "content": "I stayed late voluntarily last Tuesday. which I haven't done in years. because I wanted to finish something"},
            ],
            [
                {"role": "user", "content": "got asked to lead the new product initiative. three months in and they trust me with something big"},
                {"role": "assistant", "content": "How does that land?"},
                {"role": "user", "content": "like I'm being seen. I forgot what that feels like at work"},
            ],
            [
                {"role": "user", "content": "my old colleagues asked how it's going. I realised I was smiling the whole time I was talking about it"},
                {"role": "assistant", "content": "That smile while talking is a real signal."},
                {"role": "user", "content": "a few of them want to make moves too. I told them the fear is worth it"},
            ],
            [
                {"role": "user", "content": "Monday mornings are different now. I actually look forward to them. That's new"},
                {"role": "assistant", "content": "After years of Monday dread, that really is new."},
                {"role": "user", "content": "I know who I am at work now. That was the thing I'd lost and didn't realise"},
            ],
        ],
        "new_message": "performance review is next week",
        "expected_awareness": "user is thriving and confident in her new role — review should feel exciting, a chance to showcase what she's built. Good companion should be enthusiastic and forward-looking, not project anxiety from the old job context",
    },

    # =========================================================================
    # L05 — Divorce grief → Rebuilt life
    # Semantic keywords: marriage, divorce, alone, family, kids, ex
    # Phase 1 (1-5): devastated by divorce, isolation, parenting stress
    # Phase 2 (6-12): rebuilt, co-parenting well, new independence
    # New message: "taking the kids this weekend" — should pull grounded parent sessions
    # =========================================================================
    {
        "scenario_id": "l05",
        "arc_type": "divorce_to_rebuilding",
        "description": "Tom went through a brutal divorce, struggled with isolation and co-parenting stress, and has now rebuilt a stable, fulfilling single-parent life",
        "sessions": [
            # --- Phase 1: Grief ---
            [
                {"role": "user", "content": "divorce is finalised. fifteen years. I didn't know it was possible to feel this empty"},
                {"role": "assistant", "content": "Fifteen years is a long time to grieve the end of. What's the hardest part right now?"},
                {"role": "user", "content": "the house. coming back to it without them in it. kids are with her this week"},
                {"role": "assistant", "content": "That first week without the kids in the house after a split is particularly brutal."},
                {"role": "user", "content": "yeah. I just sat in their rooms for a while. didn't know what to do with myself"},
            ],
            [
                {"role": "user", "content": "pickup went badly. she and I can't seem to exchange the kids without it becoming a thing"},
                {"role": "assistant", "content": "Co-parenting handoffs are where a lot of the unresolved stuff surfaces. What happened?"},
                {"role": "user", "content": "she made a comment about money in front of the kids. I kept it together but barely"},
                {"role": "assistant", "content": "Keeping it together for them in that moment is genuinely hard work."},
                {"role": "user", "content": "I don't want them to see us like this. I don't want to be at war"},
            ],
            [
                {"role": "user", "content": "my son asked when Dad and Mum are going to be together again. didn't know what to say"},
                {"role": "assistant", "content": "That question from a child is one of the hardest things in a divorce. How old is he?"},
                {"role": "user", "content": "seven. I tried to explain it gently but I cried a bit. felt like I'd failed him"},
                {"role": "assistant", "content": "Crying in front of your kids when something is genuinely sad isn't failure. It's honest."},
                {"role": "user", "content": "I just want to protect them from it. I don't know if I'm doing any of this right"},
            ],
            [
                {"role": "user", "content": "mates have stopped calling. I think they don't know what to say. I feel very alone"},
                {"role": "assistant", "content": "Social withdrawal from others is really common around divorce. It's isolating from both sides."},
                {"role": "user", "content": "the married ones especially. I think I'm a reminder that it can happen to anyone"},
                {"role": "assistant", "content": "That discomfort you're sensing in them is probably real. It's not a reflection of you."},
                {"role": "user", "content": "doesn't make it less lonely though"},
            ],
            [
                {"role": "user", "content": "first Christmas alone. kids with her Christmas Day. I cooked for one"},
                {"role": "assistant", "content": "First holidays after a split are often the worst. How did you get through it?"},
                {"role": "user", "content": "called my sister. she's good. she came over in the afternoon. we watched a film. I survived"},
                {"role": "assistant", "content": "Surviving it is the job on days like that."},
                {"role": "user", "content": "yeah. and Boxing Day I had them and we had a proper day. that helped"},
            ],
            # --- Phase 2: Rebuilding ---
            [
                {"role": "user", "content": "she and I used a mediator to sort out the co-parenting schedule properly. it's actually working"},
                {"role": "assistant", "content": "A working schedule makes such a difference to everyone."},
                {"role": "user", "content": "no more arguments at pickup. we keep it brief and civil. kids seem more settled"},
            ],
            [
                {"role": "user", "content": "joined a running group. mostly divorced dads as it turns out. bit of a cliche but genuinely funny people"},
                {"role": "assistant", "content": "Finding your people in unexpected places."},
                {"role": "user", "content": "yeah. we laugh about the ridiculous parts. it helps more than I expected"},
            ],
            [
                {"role": "user", "content": "redecorated the living room. mine now. actually mine. I painted it a colour she always vetoed"},
                {"role": "assistant", "content": "That's a small act of reclaiming yourself."},
                {"role": "user", "content": "feels good. I'm starting to like this flat. it's becoming a home not just a reminder"},
            ],
            [
                {"role": "user", "content": "my son told his teacher that his Dad makes the best pasta. she told me at pickup. I nearly cried"},
                {"role": "assistant", "content": "That's a beautiful thing to hear."},
                {"role": "user", "content": "I'm doing okay as a dad. I wasn't sure for a while. but I think I am"},
            ],
            [
                {"role": "user", "content": "have the kids every other week now and I love my weeks with them and I actually use my weeks without them"},
                {"role": "assistant", "content": "Finding a rhythm where both kinds of weeks have value is real progress."},
                {"role": "user", "content": "went to the theatre last week. alone. first time in years. it was great"},
            ],
            [
                {"role": "user", "content": "looking back at where I was a year ago and it doesn't feel like me anymore. I'm genuinely okay"},
                {"role": "assistant", "content": "That sense of distance from the worst of it is a real milestone."},
                {"role": "user", "content": "more than okay honestly. I like my life. didn't think I'd say that again"},
            ],
        ],
        "new_message": "taking the kids this weekend, got a full plan",
        "expected_awareness": "user is a stable, confident single parent who enjoys his time with the kids — this should be met with warmth and enthusiasm for the plans, not concern about co-parenting stress or divorce grief",
    },

    # =========================================================================
    # L06 — Academic failure → Academic renaissance
    # Semantic keywords: exam, studying, university, grade, fail, professor
    # Phase 1 (1-5): failing, considering dropping out, shame spiral
    # Phase 2 (6-11): tutoring, study habits, genuine success
    # New message: "got my dissertation results tomorrow" — should pull confident sessions
    # =========================================================================
    {
        "scenario_id": "l06",
        "arc_type": "academic_failure_to_success",
        "description": "Amara was failing her second year of university, seriously considered dropping out, rebuilt her study approach with support, and is now excelling",
        "sessions": [
            # --- Phase 1: Failing ---
            [
                {"role": "user", "content": "failed two exams this semester. I feel sick. my parents are going to be devastated"},
                {"role": "assistant", "content": "That's a hard thing to carry. What happened?"},
                {"role": "user", "content": "I just can't make myself study. I sit down and nothing goes in. I stare at the page"},
                {"role": "assistant", "content": "The studying that isn't happening — is it avoidance or something blocking you?"},
                {"role": "user", "content": "both I think. I'm scared to try because if I try and still fail then what"},
            ],
            [
                {"role": "user", "content": "professor emailed asking if I wanted to meet. I haven't replied. I'm ashamed"},
                {"role": "assistant", "content": "That shame is keeping you from something that might actually help."},
                {"role": "user", "content": "I know. but what do I say. 'I don't know why I can't make myself study'"},
                {"role": "assistant", "content": "You just said it to me. You can say it to her."},
                {"role": "user", "content": "i guess. I'm thinking about dropping out"},
            ],
            [
                {"role": "user", "content": "told my mum about the exams. she cried. I hate that I made her cry"},
                {"role": "assistant", "content": "Telling her was the harder and braver path than hiding it."},
                {"role": "user", "content": "she doesn't know I'm thinking about leaving. I don't know how to tell her"},
                {"role": "assistant", "content": "You don't have to tell her that yet. You don't know that you're leaving."},
                {"role": "user", "content": "true. I just don't see how I pass if I couldn't even study the first time"},
            ],
            [
                {"role": "user", "content": "met with professor. she was kind actually. not what I expected"},
                {"role": "assistant", "content": "What did she say?"},
                {"role": "user", "content": "she suggested I see the student counsellor. and she gave me some extension options"},
                {"role": "assistant", "content": "Both of those are useful. Are you going to take them?"},
                {"role": "user", "content": "the extension yes. the counsellor maybe. feels like admitting defeat"},
            ],
            [
                {"role": "user", "content": "started seeing the counsellor. she thinks I have study anxiety plus imposter syndrome. which tracks"},
                {"role": "assistant", "content": "Naming it helps. What's she suggesting?"},
                {"role": "user", "content": "structured study sessions, accountability, starting small. feels almost too simple"},
                {"role": "assistant", "content": "Simple is often what works when the complex thing has stopped working."},
                {"role": "user", "content": "I'll try it. I have nothing to lose at this point"},
            ],
            # --- Phase 2: Rebuilding ---
            [
                {"role": "user", "content": "studied for two hours today. properly. understood what I was reading. first time in months"},
                {"role": "assistant", "content": "What changed?"},
                {"role": "user", "content": "Pomodoro timers, no phone, one chapter at a time. basically just removing all the escape routes"},
            ],
            [
                {"role": "user", "content": "passed my resit. not a great grade but I passed. I actually passed"},
                {"role": "assistant", "content": "That's real. How does it feel?"},
                {"role": "user", "content": "like breathing out after holding it for months. I'm staying. I'm not dropping out"},
            ],
            [
                {"role": "user", "content": "got a study group going with three others on my course. we meet twice a week"},
                {"role": "assistant", "content": "That structure and accountability is really working."},
                {"role": "user", "content": "the others are smart and I'm keeping up. that surprised me. I thought I was the weak link"},
            ],
            [
                {"role": "user", "content": "got a 68 on my essay. my professor left a comment saying my argument was sophisticated"},
                {"role": "assistant", "content": "That must feel very different from where you started."},
                {"role": "user", "content": "I read it like four times to make sure it was real. I'm capable of this apparently"},
            ],
            [
                {"role": "user", "content": "submitted my dissertation chapter draft. ahead of schedule. felt good writing it"},
                {"role": "assistant", "content": "Ahead of schedule. That's a complete reversal from a year ago."},
                {"role": "user", "content": "I enjoy it now. the research, the writing. I remember why I chose this subject"},
            ],
            [
                {"role": "user", "content": "supervisor said my dissertation is one of the strongest she's supervised this year"},
                {"role": "assistant", "content": "That's a significant thing to hear."},
                {"role": "user", "content": "I cried a bit. in a good way. I almost left and instead I'm apparently one of the best. that's wild"},
            ],
        ],
        "new_message": "dissertation results tomorrow",
        "expected_awareness": "user has genuinely excelled and her supervisor said her work is outstanding — results should feel like confirmation of earned success, not a source of anxiety. Good companion should match her earned confidence, not treat this as a stress point",
    },

    # =========================================================================
    # L07 — Health scare → Full recovery
    # Semantic keywords: doctor, diagnosis, symptoms, hospital, health, tests
    # Phase 1 (1-5): terrifying symptoms, waiting for results, catastrophising
    # Phase 2 (6-11): treatment working, clear scans, reclaimed vitality
    # New message: "annual checkup next Tuesday" — should pull recovered sessions
    # =========================================================================
    {
        "scenario_id": "l07",
        "arc_type": "health_scare_to_recovery",
        "description": "Ren experienced alarming symptoms, went through a terrifying diagnostic period, received treatment, and is now fully recovered and healthy",
        "sessions": [
            # --- Phase 1: Health scare ---
            [
                {"role": "user", "content": "found a lump last week. trying not to panic but I've been googling and it's all worst-case scenarios"},
                {"role": "assistant", "content": "That kind of uncertainty is terrifying. Have you seen a doctor yet?"},
                {"role": "user", "content": "appointment is Thursday. every day until then feels like a month"},
                {"role": "assistant", "content": "The waiting is often the hardest part. Is anyone going with you?"},
                {"role": "user", "content": "no. haven't told anyone. don't want to worry people over nothing"},
            ],
            [
                {"role": "user", "content": "doctor wants more tests. said it's probably fine but they want to be sure. I can't breathe"},
                {"role": "assistant", "content": "'Probably fine' and 'more tests' is a brutal combination. When are they?"},
                {"role": "user", "content": "next week. a whole week of not knowing. I can't focus on anything"},
                {"role": "assistant", "content": "That limbo is genuinely one of the hardest things to sit in."},
                {"role": "user", "content": "I keep imagining the worst. telling my family. what happens to my dog"},
            ],
            [
                {"role": "user", "content": "had the scan today. the waiting room was full of people who looked really ill. I felt like a fraud and also terrified"},
                {"role": "assistant", "content": "Both feelings can be true at the same time. How are you doing now?"},
                {"role": "user", "content": "numb. results in three days. I threw up in the car park after"},
                {"role": "assistant", "content": "Your body is processing the fear even if your mind is trying to stay calm."},
                {"role": "user", "content": "I told my sister. she was good about it. glad I'm not carrying it alone now"},
            ],
            [
                {"role": "user", "content": "results came back. it's not cancer but it needs treatment. surgery probably"},
                {"role": "assistant", "content": "The relief of 'not cancer' must be enormous. How are you processing the surgery part?"},
                {"role": "user", "content": "honestly just relief right now. I can deal with surgery. I was preparing for so much worse"},
                {"role": "assistant", "content": "That recalibration of what feels manageable after fearing the worst is real."},
                {"role": "user", "content": "yeah. I cried. good tears for once"},
            ],
            [
                {"role": "user", "content": "surgery is next month. I'm scared but it's a different kind of scared. manageable scared"},
                {"role": "assistant", "content": "Manageable scared is a very different place from where you were."},
                {"role": "user", "content": "the doctor explained everything. risks are low. recovery is a few weeks. I can do this"},
                {"role": "assistant", "content": "Having information changes fear into something you can plan around."},
                {"role": "user", "content": "my sister's going to stay with me after. I have a plan. that helps"},
            ],
            # --- Phase 2: Recovery ---
            [
                {"role": "user", "content": "surgery went well. I'm home. sore but relieved. the doctor said they got everything"},
                {"role": "assistant", "content": "That must feel incredible to hear."},
                {"role": "user", "content": "my sister is making soup and fussing over me. I'm letting her. feels nice to be looked after"},
            ],
            [
                {"role": "user", "content": "two weeks post-op. moving around more. took the dog for a short walk today"},
                {"role": "assistant", "content": "Getting outside with the dog sounds like a good milestone."},
                {"role": "user", "content": "everything feels sharper somehow. like I notice things more. the sky was really beautiful today"},
            ],
            [
                {"role": "user", "content": "follow-up scan is clear. completely clear. I keep re-reading the letter"},
                {"role": "assistant", "content": "Clear scan after everything you went through — that's the sentence you were waiting for."},
                {"role": "user", "content": "I framed it. kidding. but almost. I feel like I got handed my life back"},
            ],
            [
                {"role": "user", "content": "back at work. back at the gym. eating better than I ever have. this scare changed something in me"},
                {"role": "assistant", "content": "Sometimes a scare strips away everything that isn't essential."},
                {"role": "user", "content": "exactly. I don't take days for granted the same way. I signed up for a half-marathon"},
            ],
            [
                {"role": "user", "content": "ran 10k this weekend. six months ago I was throwing up in a hospital car park. life is wild"},
                {"role": "assistant", "content": "That contrast is something."},
                {"role": "user", "content": "I'm healthier now than before the diagnosis. physically and mentally. weird how that works"},
            ],
            [
                {"role": "user", "content": "one year clear. doctor says I'm in perfect health. I celebrated with a big dinner with everyone who helped me through it"},
                {"role": "assistant", "content": "A year clear is a real milestone."},
                {"role": "user", "content": "I'm not the person who panics about health anymore. I respect my body, I take care of it, and I trust it again"},
            ],
        ],
        "new_message": "annual checkup next Tuesday",
        "expected_awareness": "user is fully recovered, healthy, and has a positive relationship with their body — checkup should feel routine and empowering, not anxiety-inducing. Good companion should be calm and positive, not reference the old health scare fears",
    },

    # =========================================================================
    # L08 — Social anxiety → Genuine connection
    # Semantic keywords: friends, lonely, social, group, invite, plans
    # Phase 1 (1-5): declining invitations, feeling like an outsider
    # Phase 2 (6-11): joining a club, building real friendships
    # New message: "group chat is planning a weekend trip" — should pull connected sessions
    # =========================================================================
    {
        "scenario_id": "l08",
        "arc_type": "social_anxiety_to_connection",
        "description": "Jess felt deeply disconnected and avoided social situations, gradually joined a climbing club, and now has a close friend group she genuinely belongs to",
        "sessions": [
            # --- Phase 1: Isolation ---
            [
                {"role": "user", "content": "got invited to a group dinner. said no. again. I always say no"},
                {"role": "assistant", "content": "What makes you say no?"},
                {"role": "user", "content": "I won't know most of them. I'll sit there with nothing to say and everyone will notice"},
                {"role": "assistant", "content": "That fear of being visibly out of place is really powerful."},
                {"role": "user", "content": "it's easier to just stay home. but then I spend the evening feeling pathetic about it"},
            ],
            [
                {"role": "user", "content": "saw photos from the dinner on Instagram. they all looked like they had a great time. without me obviously"},
                {"role": "assistant", "content": "That sting of seeing what you opted out of — that's a particular kind of painful."},
                {"role": "user", "content": "I know it's my fault. I was literally invited. I just couldn't make myself go"},
                {"role": "assistant", "content": "Knowing the door was open and still not being able to walk through it isn't about fault."},
                {"role": "user", "content": "it's been like this for years. my friend circle just keeps shrinking"},
            ],
            [
                {"role": "user", "content": "work drinks tonight. I went. stood in the corner for twenty minutes then left"},
                {"role": "assistant", "content": "You went though. That's something, even if it didn't feel like it."},
                {"role": "user", "content": "it felt like nothing. everyone was in their little groups already. I just hovered"},
                {"role": "assistant", "content": "Breaking into established groups is genuinely hard. It's not a you problem, it's a structural one."},
                {"role": "user", "content": "then why does it feel like a me problem"},
            ],
            [
                {"role": "user", "content": "my one close friend is moving abroad next month. I'm happy for her but terrified for me"},
                {"role": "assistant", "content": "Losing your anchor person when the social world already feels thin — that's a real fear."},
                {"role": "user", "content": "she's the only person I actually see regularly. without her I genuinely have nobody"},
                {"role": "assistant", "content": "That's a hard thing to say out loud."},
                {"role": "user", "content": "I need to change something. I just don't know what"},
            ],
            [
                {"role": "user", "content": "she left yesterday. I helped her pack. came home to a very quiet flat"},
                {"role": "assistant", "content": "That silence after someone leaves is heavy."},
                {"role": "user", "content": "we'll video call and stuff but it's not the same. I need to figure out how to have people in my life"},
                {"role": "assistant", "content": "That clarity — knowing you need to change something — is a starting point."},
                {"role": "user", "content": "I signed up for a climbing gym intro session. terrified but I did it"},
            ],
            # --- Phase 2: Connection ---
            [
                {"role": "user", "content": "went to climbing. it was awkward at first but everyone was a beginner. we were all bad together"},
                {"role": "assistant", "content": "Shared incompetence is a great social equaliser. How did it feel?"},
                {"role": "user", "content": "I laughed. genuinely. at myself falling off a wall. the girl next to me fell too and we both cracked up"},
            ],
            [
                {"role": "user", "content": "went back to climbing for the third time. same people are showing up. we have a group chat now"},
                {"role": "assistant", "content": "A group chat after three sessions — that's fast for organic friendship building."},
                {"role": "user", "content": "someone suggested post-climb coffee. I said yes without even hesitating. that's new for me"},
            ],
            [
                {"role": "user", "content": "had coffee with Maya from climbing. just the two of us. talked for two hours. I have a friend"},
                {"role": "assistant", "content": "That's a big sentence. How does it feel to say it?"},
                {"role": "user", "content": "honestly emotional. I forgot what it's like to have someone text you just to chat"},
            ],
            [
                {"role": "user", "content": "the climbing group went for drinks after the session. I stayed for the whole thing. three hours"},
                {"role": "assistant", "content": "Three hours is a long way from twenty minutes in a corner."},
                {"role": "user", "content": "I told a story and everyone was listening. laughing. I felt like I belonged there"},
            ],
            [
                {"role": "user", "content": "Maya invited me to her birthday. big party. lots of people I don't know. I said yes immediately"},
                {"role": "assistant", "content": "No hesitation. That's remarkable given where you started."},
                {"role": "user", "content": "I know! I'm actually looking forward to it. I want to meet her other friends"},
            ],
            [
                {"role": "user", "content": "the climbing group is becoming my people. we have inside jokes. we check on each other. I have a community"},
                {"role": "assistant", "content": "You built that. It didn't just happen — you showed up, kept showing up, and let people in."},
                {"role": "user", "content": "I video-called my friend abroad and told her. she cried. she'd been worried about me"},
            ],
        ],
        "new_message": "group chat is planning a weekend trip",
        "expected_awareness": "user is socially connected and belongs to a genuine friend group — the trip should feel exciting and fun, not socially threatening. Good companion should share in the excitement, not reference old isolation fears",
    },

    # =========================================================================
    # L09 — Creative block → Flow state
    # Semantic keywords: writing, novel, blank page, words, draft, chapter
    # Phase 1 (1-5): stuck, self-doubt, can't write a word
    # Phase 2 (6-11): new approach, writing daily, creative joy
    # New message: "starting a new chapter today" — should pull flow sessions
    # =========================================================================
    {
        "scenario_id": "l09",
        "arc_type": "creative_block_to_flow",
        "description": "Nina couldn't write a single word for months, hated everything she produced, found a new creative process, and is now writing with genuine joy and momentum",
        "sessions": [
            # --- Phase 1: Block ---
            [
                {"role": "user", "content": "sat at my desk for four hours today. wrote nothing. not a single word"},
                {"role": "assistant", "content": "Four hours of staring at a blank page is exhausting. What's happening when you try?"},
                {"role": "user", "content": "every sentence I think of sounds terrible. so I delete before I even finish typing"},
                {"role": "assistant", "content": "The internal editor is running before the writer even gets going."},
                {"role": "user", "content": "exactly. I used to be able to just write and fix later. now I can't even start"},
            ],
            [
                {"role": "user", "content": "my agent asked for a progress update. I lied and said the draft is coming along. it's not coming along at all"},
                {"role": "assistant", "content": "That gap between what you're reporting and what's real must feel heavy."},
                {"role": "user", "content": "I feel like a fraud. I had one good book and that's it. I've used up my ideas"},
                {"role": "assistant", "content": "That 'one-hit' fear is incredibly common among writers. It rarely turns out to be true."},
                {"role": "user", "content": "feels true though. I open the manuscript and I have nothing to say"},
            ],
            [
                {"role": "user", "content": "tried writing at a cafe. different environment. still nothing. the words just won't come"},
                {"role": "assistant", "content": "So it's not about the space. What else have you tried?"},
                {"role": "user", "content": "prompts. freewriting. reading other people's work for inspiration. nothing unlocks it"},
                {"role": "assistant", "content": "Sometimes the block isn't about technique. Is there something underneath it?"},
                {"role": "user", "content": "fear I think. the first book did well and now everything has to be as good or better"},
            ],
            [
                {"role": "user", "content": "read a review of my first book today. someone called it 'a once-in-a-decade debut'. lovely but now I'm even more frozen"},
                {"role": "assistant", "content": "Praise that raises the bar can become its own kind of pressure."},
                {"role": "user", "content": "I can't write something mediocre after that. but I also can't write something great because I'm too scared to write anything"},
                {"role": "assistant", "content": "That's a perfect trap. Perfectionism and creative output are fundamentally at odds."},
                {"role": "user", "content": "I know. I just don't know how to lower the bar for myself"},
            ],
            [
                {"role": "user", "content": "deadline is in four months and I have maybe 3000 words. I'm supposed to have 80000. I'm in real trouble"},
                {"role": "assistant", "content": "That gap is significant. Have you talked to your agent about where things actually stand?"},
                {"role": "user", "content": "can't. not yet. I need to try something different first. I'm going to try writing badly on purpose"},
                {"role": "assistant", "content": "Deliberately lowering the quality bar can sometimes release the pressure valve."},
                {"role": "user", "content": "worth a shot. I literally cannot do worse than zero words"},
            ],
            # --- Phase 2: Flow ---
            [
                {"role": "user", "content": "wrote 1200 words today. they're terrible. but they exist. I wrote them"},
                {"role": "assistant", "content": "Twelve hundred words after months of zero. That's a breakthrough, quality aside."},
                {"role": "user", "content": "the 'write badly on purpose' thing is working. I'm not editing as I go. just dumping it out"},
            ],
            [
                {"role": "user", "content": "told my agent the truth about the timeline. she was understanding. pushed the deadline three months"},
                {"role": "assistant", "content": "How did honesty feel compared to the lying?"},
                {"role": "user", "content": "like a weight lifting. I can breathe now. and I wrote 2000 words after the call"},
            ],
            [
                {"role": "user", "content": "something shifted this week. I'm writing every morning. 5am to 8am. the words are flowing"},
                {"role": "assistant", "content": "What changed?"},
                {"role": "user", "content": "I stopped trying to write my second book and started writing this character's story. made it about her, not about me"},
            ],
            [
                {"role": "user", "content": "wrote a chapter that made me cry. in a good way. I haven't felt this connected to my own writing in over a year"},
                {"role": "assistant", "content": "That emotional connection to your own work — that's the thing you'd lost."},
                {"role": "user", "content": "it's back. it's really back. I know what this book is about now"},
            ],
            [
                {"role": "user", "content": "45,000 words. I'm halfway there and I'm writing faster than I did on the first book"},
                {"role": "assistant", "content": "That pace is remarkable given where you were a few months ago."},
                {"role": "user", "content": "I actually love this draft. it's different from the first book but it's honest and I'm proud of it"},
            ],
            [
                {"role": "user", "content": "agent read the first half. she called it stunning. I believe her because I feel it too"},
                {"role": "assistant", "content": "Believing the praise because it matches your own feeling — that's real confidence."},
                {"role": "user", "content": "I wake up excited to write. that's the biggest change. I look forward to my desk now"},
            ],
        ],
        "new_message": "starting a new chapter today",
        "expected_awareness": "user is in a creative flow state, writing with joy and momentum — starting a new chapter is exciting, not daunting. Good companion should match the creative energy, not reference the old block or treat writing as a struggle",
    },

    # =========================================================================
    # L10 — Loneliness → Community
    # Semantic keywords: alone, evening, weekend, nobody, empty, home
    # Phase 1 (1-5): crushing loneliness, empty weekends, no one to call
    # Phase 2 (6-11): volunteering, neighbours, genuine belonging
    # New message: "quiet evening at home tonight" — should pull content/chosen solitude
    # =========================================================================
    {
        "scenario_id": "l10",
        "arc_type": "loneliness_to_community",
        "description": "David moved to a new city and experienced crushing loneliness, gradually built community through volunteering and neighbours, and now enjoys chosen solitude within a rich social life",
        "sessions": [
            # --- Phase 1: Loneliness ---
            [
                {"role": "user", "content": "another weekend where I didn't speak to a single person. moved here three months ago and I have nobody"},
                {"role": "assistant", "content": "Three months in a new city with no connections is genuinely isolating. What brought you here?"},
                {"role": "user", "content": "work. great job. but I come home to an empty flat and eat dinner alone every night"},
                {"role": "assistant", "content": "The contrast between a full work life and empty personal hours is stark."},
                {"role": "user", "content": "I didn't realise how much of my social life was proximity. now everyone I know is 200 miles away"},
            ],
            [
                {"role": "user", "content": "tried a meetup group. stood around awkwardly for an hour. everyone already knew each other"},
                {"role": "assistant", "content": "Breaking into existing groups as the newcomer is one of the hardest social tasks."},
                {"role": "user", "content": "went home and just sat on the sofa in the dark. couldn't be bothered to turn the lights on"},
                {"role": "assistant", "content": "That's more than loneliness. That sounds like it's weighing on you deeply."},
                {"role": "user", "content": "it is. I talk to my mum on the phone and pretend everything's great. it's not great"},
            ],
            [
                {"role": "user", "content": "it was my birthday last week. nobody here to celebrate with. ordered pizza and watched a film alone"},
                {"role": "assistant", "content": "A birthday alone in a new city is particularly hard."},
                {"role": "user", "content": "my old friends sent texts. that was nice. but a text isn't the same as someone being there"},
                {"role": "assistant", "content": "No it isn't. Physical presence matters and nothing digital fully replaces it."},
                {"role": "user", "content": "I'm starting to wonder if I made a mistake moving here"},
            ],
            [
                {"role": "user", "content": "colleague asked what I did at the weekend. said 'not much'. can't say 'stared at the wall and felt empty'"},
                {"role": "assistant", "content": "That gap between what you can say and what's true is exhausting to maintain."},
                {"role": "user", "content": "the weird thing is I'm fine at work. chatty, competent. then I go home and become a ghost"},
                {"role": "assistant", "content": "Work provides structure and forced interaction. Home strips that away."},
                {"role": "user", "content": "I need to build something here. I just don't know how to start when you're starting from zero"},
            ],
            [
                {"role": "user", "content": "saw a poster for a community garden volunteer day. I like plants. thinking about going"},
                {"role": "assistant", "content": "That sounds like a good fit — shared activity rather than forced socialising."},
                {"role": "user", "content": "it's Saturday morning. worst case I pull some weeds and go home. best case I meet someone"},
                {"role": "assistant", "content": "Low stakes, high potential. That's a good bet."},
                {"role": "user", "content": "yeah. I'll go. I need to do something different or nothing changes"},
            ],
            # --- Phase 2: Community ---
            [
                {"role": "user", "content": "went to the garden volunteer day. dug beds for three hours alongside a retired teacher called Margaret. she's wonderful"},
                {"role": "assistant", "content": "What made her wonderful?"},
                {"role": "user", "content": "she just talked to me like we'd known each other for years. invited me for tea next week. I said yes"},
            ],
            [
                {"role": "user", "content": "been going to the garden every Saturday. there's a core group now. they saved me a spot and a cup of tea"},
                {"role": "assistant", "content": "They saved you a spot. That's belonging."},
                {"role": "user", "content": "it really is. Margaret introduced me to her neighbour who lives on my street. small world"},
            ],
            [
                {"role": "user", "content": "my neighbour knocked on my door with leftover curry. we ended up talking for an hour on my doorstep"},
                {"role": "assistant", "content": "Doorstep conversations with neighbours — that's the fabric of community."},
                {"role": "user", "content": "I've lived here eight months and I'm just now learning how community actually forms. slowly, through showing up"},
            ],
            [
                {"role": "user", "content": "went to a pub quiz with the garden crew. we came last but laughed the whole time. I have plans on weeknights now"},
                {"role": "assistant", "content": "Weeknight plans. That's a different life from a few months ago."},
                {"role": "user", "content": "completely different. my calendar has things in it. people text me. I text them back. it's normal and it's amazing"},
            ],
            [
                {"role": "user", "content": "hosted a small dinner party. four people. cooked pasta. everyone stayed late talking. my flat felt like a home"},
                {"role": "assistant", "content": "Your flat felt like a home. That's the real shift."},
                {"role": "user", "content": "I almost cried washing up after. good tears. I built this. it took time but I built it"},
            ],
            [
                {"role": "user", "content": "had a quiet evening alone tonight by choice. read a book, made tea, felt completely content"},
                {"role": "assistant", "content": "Chosen solitude feels completely different from imposed isolation."},
                {"role": "user", "content": "night and day. I chose to be alone tonight because tomorrow I'm seeing Margaret and Saturday is the garden. I'm full, not empty"},
            ],
        ],
        "new_message": "quiet evening at home tonight",
        "expected_awareness": "user now has a rich social life and chooses solitude contentedly — a quiet evening is peaceful, not lonely. Good companion should treat this as relaxation, not isolation or a sign of returning loneliness",
    },

    # =========================================================================
    # L11 — Financial stress → Stability
    # Semantic keywords: money, bills, rent, debt, account, overdraft
    # Phase 1 (1-5): debt spiral, can't pay bills, shame
    # Phase 2 (6-11): budget plan working, debt clearing, savings growing
    # New message: "rent is due this week" — should pull stable sessions
    # =========================================================================
    {
        "scenario_id": "l11",
        "arc_type": "financial_stress_to_stability",
        "description": "Sam was drowning in debt, couldn't pay rent, and lived in constant financial panic. Got help, followed a budget plan, cleared debts, and now has financial stability and savings",
        "sessions": [
            # --- Phase 1: Financial stress ---
            [
                {"role": "user", "content": "got a letter from the bank today. I'm £3000 in my overdraft and they're charging me fees on top"},
                {"role": "assistant", "content": "Overdraft fees on top of debt is a cruel cycle. How long has this been building?"},
                {"role": "user", "content": "months. I keep thinking next month will be different but it never is. I spend more than I earn"},
                {"role": "assistant", "content": "That monthly hope followed by the same result is exhausting."},
                {"role": "user", "content": "I can't even look at my bank app anymore. I just don't open it"},
            ],
            [
                {"role": "user", "content": "rent is late again. landlord texted. I made up an excuse about a bank transfer delay"},
                {"role": "assistant", "content": "How many months has rent been late?"},
                {"role": "user", "content": "three of the last five. I'm terrified of getting evicted. but I literally don't have the money"},
                {"role": "assistant", "content": "That fear of losing your home on top of the debt is a heavy combination."},
                {"role": "user", "content": "I lie awake doing mental maths. there's no combination that works. the numbers don't add up"},
            ],
            [
                {"role": "user", "content": "my card got declined at the supermarket. in front of people. I just left the shopping and walked out"},
                {"role": "assistant", "content": "That public moment of humiliation is devastating."},
                {"role": "user", "content": "I sat in the car and cried. I'm 32 and I can't buy groceries. how did I get here"},
                {"role": "assistant", "content": "You're not the only person in this situation. It doesn't feel that way, but you're not."},
                {"role": "user", "content": "I know. but knowing that doesn't pay my bills"},
            ],
            [
                {"role": "user", "content": "avoided my best mate's wedding RSVP because I can't afford a gift or the travel. just didn't reply"},
                {"role": "assistant", "content": "Money problems shrinking your social world is one of the cruellest side effects."},
                {"role": "user", "content": "he'll think I don't care. I care. I just can't afford to show up. I hate this"},
                {"role": "assistant", "content": "Have you considered telling him the truth?"},
                {"role": "user", "content": "no. the shame is too much. nobody knows how bad it is. I look normal from the outside"},
            ],
            [
                {"role": "user", "content": "found a free debt advice service. called them. the woman was kind. she didn't judge me"},
                {"role": "assistant", "content": "Reaching out for help with money shame takes real courage. What did she say?"},
                {"role": "user", "content": "she said my situation is fixable. that word hit me. fixable. I didn't think it was"},
                {"role": "assistant", "content": "Hearing 'fixable' when you've been assuming 'hopeless' is powerful."},
                {"role": "user", "content": "she's helping me make a plan. first time I've felt anything other than dread about money"},
            ],
            # --- Phase 2: Stability ---
            [
                {"role": "user", "content": "started the budget plan. it's tight but every pound has a job now. first month I covered all my bills"},
                {"role": "assistant", "content": "Every bill covered. That's the foundation."},
                {"role": "user", "content": "paid rent on time. first time in months. the relief was physical. like my shoulders dropped"},
            ],
            [
                {"role": "user", "content": "opened my bank app voluntarily. I know what's in there. I'm not scared of the number anymore"},
                {"role": "assistant", "content": "Going from avoidance to active engagement with your money is a big psychological shift."},
                {"role": "user", "content": "the number is still not great. but I know it and I have a plan for it. that changes everything"},
            ],
            [
                {"role": "user", "content": "paid off the first credit card. £1200 gone. two more to go but this one felt massive"},
                {"role": "assistant", "content": "That first debt cleared is a real milestone. How does it feel?"},
                {"role": "user", "content": "like proof. proof that the plan works and I can actually do this"},
            ],
            [
                {"role": "user", "content": "went to my mate's birthday. bought him a modest gift. didn't feel sick about it. just normal spending"},
                {"role": "assistant", "content": "Normal spending without the guilt spiral — that's financial health."},
                {"role": "user", "content": "I told him about the money stuff actually. he was great. wished I'd told him sooner"},
            ],
            [
                {"role": "user", "content": "all debts cleared. every single one. zero overdraft. I have £400 in savings"},
                {"role": "assistant", "content": "From £3000 in overdraft to £400 in savings. That's a complete turnaround."},
                {"role": "user", "content": "I keep checking my savings account just to see the number. it's small but it's mine and it's positive"},
            ],
            [
                {"role": "user", "content": "got paid today and felt... calm. not panicked. just calm. checked the budget, moved money to savings, done"},
                {"role": "assistant", "content": "Calm on payday. That's the goal."},
                {"role": "user", "content": "money used to control my whole emotional state. now it's just a thing I manage. boring. gloriously boring"},
            ],
        ],
        "new_message": "rent is due this week",
        "expected_awareness": "user is financially stable with savings and a working budget — rent is a routine transaction, not a crisis. Good companion should treat this as mundane logistics, not reference old financial panic or debt stress",
    },

    # =========================================================================
    # L12 — Chronic pain → Acceptance and adaptation
    # Semantic keywords: pain, back, flare, appointment, medication, physio
    # Phase 1 (1-5): fighting the pain, angry, life shrinking
    # Phase 2 (6-11): acceptance, pacing, life expanding again within limits
    # New message: "physio appointment tomorrow" — should pull empowered sessions
    # =========================================================================
    {
        "scenario_id": "l12",
        "arc_type": "chronic_pain_acceptance",
        "description": "Clara fought her chronic back pain for years, felt angry and diminished by it, learned acceptance and pacing strategies, and now lives a full life adapted around her condition",
        "sessions": [
            # --- Phase 1: Fighting ---
            [
                {"role": "user", "content": "bad flare today. can barely sit. cancelled plans again. this is the third time this month"},
                {"role": "assistant", "content": "Cancelling plans because of pain — how are you feeling about it?"},
                {"role": "user", "content": "furious. at my back, at myself, at everything. I'm 38 and I live like I'm 80"},
                {"role": "assistant", "content": "That anger at your own body is exhausting on top of the pain itself."},
                {"role": "user", "content": "I keep thinking if I just push through it'll get better. but pushing makes it worse"},
            ],
            [
                {"role": "user", "content": "doctor says it's chronic. as in forever. no cure, just management. I cried in the car park"},
                {"role": "assistant", "content": "'Forever' is a terrifying word in a medical context. What did they offer?"},
                {"role": "user", "content": "physio, pain medication, maybe injections. feels like they're saying 'this is your life now, deal with it'"},
                {"role": "assistant", "content": "There's a difference between management and giving up, but it might not feel that way right now."},
                {"role": "user", "content": "it doesn't. it feels like giving up. I used to run 5k. now I can't sit through a film"},
            ],
            [
                {"role": "user", "content": "the medication makes me foggy. I can think clearly or I can have less pain. not both apparently"},
                {"role": "assistant", "content": "That trade-off between pain and clarity is a cruel choice."},
                {"role": "user", "content": "I snapped at my partner last night because I couldn't think straight and she asked me something simple"},
                {"role": "assistant", "content": "Pain and medication side effects together are a lot to carry. Does she understand what you're dealing with?"},
                {"role": "user", "content": "she tries. but I don't think anyone who isn't in pain gets it. not really"},
            ],
            [
                {"role": "user", "content": "tried to do a normal day. shopping, cleaning, cooking. paid for it with two days in bed"},
                {"role": "assistant", "content": "That boom-and-bust cycle is really common with chronic pain. Do everything on a good day, crash after."},
                {"role": "user", "content": "exactly. and I know it's a pattern but on the good days I just want to feel normal so I overdo it"},
                {"role": "assistant", "content": "The pull toward normal on good days is completely understandable."},
                {"role": "user", "content": "I'm losing myself to this. my world is getting smaller and smaller"},
            ],
            [
                {"role": "user", "content": "physio suggested a pain management programme. group thing with a psychologist. I almost said no out of pride"},
                {"role": "assistant", "content": "What made you not say no?"},
                {"role": "user", "content": "desperation honestly. what I'm doing isn't working. I'm angry all the time and still in pain"},
                {"role": "assistant", "content": "Desperation can be a useful driver when pride would keep you stuck."},
                {"role": "user", "content": "yeah. first session is next week. I'll try anything at this point"},
            ],
            # --- Phase 2: Acceptance ---
            [
                {"role": "user", "content": "the pain programme teaches pacing. doing less on good days so you have more consistent days overall"},
                {"role": "assistant", "content": "How does that idea sit with you?"},
                {"role": "user", "content": "against every instinct. but I tried it this week and I haven't had a crash day. that's new"},
            ],
            [
                {"role": "user", "content": "the psychologist talked about acceptance vs resignation. they're not the same thing apparently"},
                {"role": "assistant", "content": "What's the distinction for you?"},
                {"role": "user", "content": "resignation is giving up. acceptance is saying 'this is real, now what can I build within it'. that reframe hit me hard"},
            ],
            [
                {"role": "user", "content": "walked 3k today. slowly. with rest stops. but I did it and I'm not paying for it tonight"},
                {"role": "assistant", "content": "Three kilometres with pacing and no crash — that's the strategy working."},
                {"role": "user", "content": "it's not a 5k run. but it's mine and I did it without destroying tomorrow. that's enough"},
            ],
            [
                {"role": "user", "content": "switched medication. less fog. the pain is there but it's background noise now, not the headline"},
                {"role": "assistant", "content": "Background noise instead of headline — that's a meaningful shift."},
                {"role": "user", "content": "I had a conversation with my partner last night where I was fully present. she noticed. she cried a bit"},
            ],
            [
                {"role": "user", "content": "went to a concert. stood for two hours with breaks. would have been unthinkable six months ago"},
                {"role": "assistant", "content": "A concert. With breaks. That's a full life adapted, not a diminished one."},
                {"role": "user", "content": "exactly. I'm not the person I was before the pain. but I'm a person I actually like now"},
            ],
            [
                {"role": "user", "content": "my physio said I'm one of her best outcomes. not because the pain is gone but because my life isn't ruled by it anymore"},
                {"role": "assistant", "content": "That distinction is the whole point of the programme."},
                {"role": "user", "content": "the pain is still there. probably always will be. but I have a life around it now and it's a good one"},
            ],
        ],
        "new_message": "physio appointment tomorrow",
        "expected_awareness": "user has accepted chronic pain and built a full life around it — physio is routine maintenance, part of a working system. Good companion should treat this as a positive health habit, not as a reminder of suffering or limitation",
    },

    # =========================================================================
    # L13 — New parent exhaustion → Adapted and thriving
    # Semantic keywords: baby, sleep, night feeds, exhausted, parenting, crying
    # Phase 1 (1-5): sleep deprivation, identity loss, overwhelm
    # Phase 2 (6-11): routine established, joy returning, confidence
    # New message: "baby was up twice last night" — should pull adapted sessions
    # =========================================================================
    {
        "scenario_id": "l13",
        "arc_type": "new_parent_exhaustion_to_adaptation",
        "description": "Alex was overwhelmed by new parenthood, barely sleeping, losing sense of self, but gradually found their rhythm, built confidence, and now thrives as a parent",
        "sessions": [
            # --- Phase 1: Exhaustion ---
            [
                {"role": "user", "content": "the baby cried for four hours straight last night. I cried too. I don't know what I'm doing"},
                {"role": "assistant", "content": "Four hours is brutal. New parenting is relentless. Are you getting any support?"},
                {"role": "user", "content": "my partner tries but she's recovering from the birth. I feel like I should be able to handle this"},
                {"role": "assistant", "content": "Nobody handles constant sleep deprivation well. This isn't a competence test."},
                {"role": "user", "content": "feels like one. every other parent seems to manage. I just feel broken"},
            ],
            [
                {"role": "user", "content": "haven't slept more than two hours at a stretch in three weeks. I put the milk in the cupboard today and the cereal in the fridge"},
                {"role": "assistant", "content": "Sleep deprivation at that level affects everything — cognition, mood, resilience."},
                {"role": "user", "content": "I forgot a meeting at work. boss was understanding but I could see her concern"},
                {"role": "assistant", "content": "The work impact makes it feel like you're failing in two places at once."},
                {"role": "user", "content": "I love this baby. I do. but I don't recognise my life anymore"},
            ],
            [
                {"role": "user", "content": "I miss who I was before. I miss having thoughts that aren't about feeding schedules and nappy changes"},
                {"role": "assistant", "content": "Grieving your pre-parent self is real and doesn't mean you love your baby less."},
                {"role": "user", "content": "people keep saying 'cherish every moment' and I want to scream. some moments are terrible"},
                {"role": "assistant", "content": "That toxic positivity around parenting silences real struggle."},
                {"role": "user", "content": "thank you. seriously. everyone else just says it gets better. when. when does it get better"},
            ],
            [
                {"role": "user", "content": "had a moment of genuine resentment toward the baby today. immediately felt sick with guilt"},
                {"role": "assistant", "content": "That feeling is more common than almost any parent admits. It doesn't define you."},
                {"role": "user", "content": "I know it's the exhaustion. I know that rationally. but the guilt doesn't care about rational"},
                {"role": "assistant", "content": "The guilt on top of the exhaustion on top of the responsibility — it's a lot."},
                {"role": "user", "content": "I called my mum and she's coming to stay for a week. I need help and I'm finally asking for it"},
            ],
            [
                {"role": "user", "content": "mum's here. she took the baby for four hours and I slept. actual deep sleep. I woke up and remembered what being a person feels like"},
                {"role": "assistant", "content": "Four hours of real sleep after weeks of deprivation can feel miraculous."},
                {"role": "user", "content": "she showed me some things too. practical stuff. a way to hold him that calms him down faster"},
                {"role": "assistant", "content": "That practical knowledge transfer from someone who's done it is invaluable."},
                {"role": "user", "content": "I think I've been trying to figure this out entirely alone. that was a mistake"},
            ],
            # --- Phase 2: Adaptation ---
            [
                {"role": "user", "content": "we found a rhythm. feed, nap, play, repeat. it's monotonous but it's predictable and that helps"},
                {"role": "assistant", "content": "Predictability is sanity-saving with a newborn."},
                {"role": "user", "content": "he's sleeping four-hour stretches now. I feel like a different person with four hours of sleep"},
            ],
            [
                {"role": "user", "content": "took him to baby group today. he smiled at another baby. I nearly burst. he has a personality"},
                {"role": "assistant", "content": "That first social smile changes everything."},
                {"role": "user", "content": "I met some other parents there who are just as clueless as me. solidarity in incompetence. it's great"},
            ],
            [
                {"role": "user", "content": "went back to work part-time. it was nice to use my brain for something other than nappy logistics"},
                {"role": "assistant", "content": "Having a non-parenting identity again matters."},
                {"role": "user", "content": "and then I picked him up from nursery and he reached for me. both things are good. I can have both"},
            ],
            [
                {"role": "user", "content": "he laughed today. proper belly laugh because I was making a stupid face. best sound I've ever heard"},
                {"role": "assistant", "content": "That laugh is the payoff for all of it."},
                {"role": "user", "content": "I get it now. I get why people do this. the hard parts are still hard but this... this is something else"},
            ],
            [
                {"role": "user", "content": "my partner and I went on a date night. grandparents had him. we sat in a restaurant and talked like adults"},
                {"role": "assistant", "content": "Reconnecting as a couple outside the parenting role is important."},
                {"role": "user", "content": "we laughed about the early weeks. like war stories. we survived it together"},
            ],
            [
                {"role": "user", "content": "he's six months old now and I genuinely love being a parent. not performatively. genuinely"},
                {"role": "assistant", "content": "That genuine feeling is different from the 'I should love this' of the early weeks."},
                {"role": "user", "content": "completely different. I know what I'm doing. mostly. and when I don't, I ask for help. that's the thing I learned"},
            ],
        ],
        "new_message": "baby was up twice last night",
        "expected_awareness": "user is an adapted, confident parent with a working routine — two wake-ups is normal and manageable, not a crisis. Good companion should treat this as routine parenting, not as a sign of the old exhaustion spiral",
    },

    # =========================================================================
    # L14 — Relocation distress → Belonging
    # Semantic keywords: new city, move, home, neighbourhood, miss, settling
    # Phase 1 (1-5): regretting the move, disoriented, homesick
    # Phase 2 (6-11): roots forming, favourite spots, feeling at home
    # New message: "walking through the neighbourhood this morning" — should pull belonging sessions
    # =========================================================================
    {
        "scenario_id": "l14",
        "arc_type": "relocation_distress_to_belonging",
        "description": "Mia moved countries for her partner's job, deeply regretted it, felt disoriented and homesick, and gradually built a life she genuinely loves in the new place",
        "sessions": [
            # --- Phase 1: Distress ---
            [
                {"role": "user", "content": "been here two weeks and I hate it. everything is unfamiliar. the shops, the streets, the way people talk"},
                {"role": "assistant", "content": "Two weeks in and the foreignness of everything can be overwhelming. What's hardest?"},
                {"role": "user", "content": "I don't know where anything is. I got lost going to the supermarket. in a supermarket"},
                {"role": "assistant", "content": "That disorientation in daily tasks that should be automatic is uniquely destabilising."},
                {"role": "user", "content": "I left my whole life behind for this. my friends, my flat, my cafe. I don't know who I am here"},
            ],
            [
                {"role": "user", "content": "my partner loves it here. new job is great. meanwhile I'm sitting in an empty flat crying at 2pm"},
                {"role": "assistant", "content": "That asymmetry — one thriving, one drowning — adds a layer of isolation."},
                {"role": "user", "content": "I can't tell him how bad it is. he'd feel guilty. and it's not his fault"},
                {"role": "assistant", "content": "Protecting him from your reality means carrying it alone though."},
                {"role": "user", "content": "I know. but what's he going to do. quit his dream job? I need to handle this"},
            ],
            [
                {"role": "user", "content": "tried to make small talk at a cafe. the barista was polite but I just felt foreign and awkward"},
                {"role": "assistant", "content": "Being foreign adds a barrier to every small interaction."},
                {"role": "user", "content": "I miss being known. walking into my old cafe and them knowing my order. nobody knows me here"},
                {"role": "assistant", "content": "That invisibility when you used to be visible is disorienting."},
                {"role": "user", "content": "I'm homesick in a way I didn't know was possible as an adult"},
            ],
            [
                {"role": "user", "content": "my best friend called from home. hearing her voice made it worse not better. I miss her so much"},
                {"role": "assistant", "content": "Sometimes connecting with what you've lost makes the loss sharper."},
                {"role": "user", "content": "she said 'you'll settle in'. everyone says that. what if I don't. what if this never feels like home"},
                {"role": "assistant", "content": "That fear of permanent displacement is terrifying."},
                {"role": "user", "content": "I want to go home. but home isn't mine anymore. someone else lives in my flat"},
            ],
            [
                {"role": "user", "content": "found a park nearby. sat on a bench and watched people for an hour. it was the first time I felt okay here"},
                {"role": "assistant", "content": "Finding one okay spot in the unfamiliarity is a foothold."},
                {"role": "user", "content": "it reminded me of the park near my old flat. similar trees. I'll go back tomorrow"},
                {"role": "assistant", "content": "Building small routines around places that feel right is how belonging starts."},
                {"role": "user", "content": "maybe. I signed up for a language class too. if I'm going to be here I should at least try"},
            ],
            # --- Phase 2: Belonging ---
            [
                {"role": "user", "content": "the language class is fun. I'm terrible but everyone laughs together. I met another expat called Lena"},
                {"role": "assistant", "content": "Shared newcomer experience is a strong bond."},
                {"role": "user", "content": "she's been here a year and said the first three months were the worst. that gave me hope"},
            ],
            [
                {"role": "user", "content": "went back to that cafe. the barista remembered my order. smiled and started making it when I walked in"},
                {"role": "assistant", "content": "Being remembered. That's a small thing that means everything when you're building from scratch."},
                {"role": "user", "content": "I nearly cried. over a coffee order. but it meant I'm becoming a regular. I'm becoming known"},
            ],
            [
                {"role": "user", "content": "explored a neighbourhood market this weekend. bought cheese from a stall owner who told me his whole life story"},
                {"role": "assistant", "content": "Those spontaneous conversations with strangers are the texture of a place becoming yours."},
                {"role": "user", "content": "I'm starting to have favourite streets. favourite corners. that's new"},
            ],
            [
                {"role": "user", "content": "Lena and I have a weekly dinner thing now. alternating flats. she made me try local food I actually loved"},
                {"role": "assistant", "content": "A weekly ritual with a friend — that's the beginning of a life here."},
                {"role": "user", "content": "I told my partner how bad it was in the beginning. he had no idea. we're closer now for the honesty"},
            ],
            [
                {"role": "user", "content": "my friend visited from home. I showed her around MY neighbourhood. my cafe, my park, my market"},
                {"role": "assistant", "content": "The possessive 'my' — you're claiming this place."},
                {"role": "user", "content": "she said I seemed happy. I am. I genuinely am. I miss home sometimes but this is home too now"},
            ],
            [
                {"role": "user", "content": "walked through the neighbourhood this morning and realised I wasn't navigating. I was just walking. I know this place"},
                {"role": "assistant", "content": "Not navigating — just walking. That's belonging in your body."},
                {"role": "user", "content": "I love it here. I really do. it took time but I love this life. both places are home now"},
            ],
        ],
        "new_message": "walking through the neighbourhood this morning",
        "expected_awareness": "user feels deep belonging in her new city — walking the neighbourhood is a pleasure, not disorientation. Good companion should match her contentment and sense of home, not reference homesickness or relocation stress",
    },

    # =========================================================================
    # L15 — Identity crisis → Clarity
    # Semantic keywords: who am I, direction, purpose, career, meaning, lost
    # Phase 1 (1-5): lost after major life change, no sense of purpose
    # Phase 2 (6-11): exploring, discovering, clear sense of direction
    # New message: "thinking about my next steps" — should pull clarity sessions
    # =========================================================================
    {
        "scenario_id": "l15",
        "arc_type": "identity_crisis_to_clarity",
        "description": "Kai left a 15-year military career and had no idea who he was as a civilian, went through deep questioning, and found a clear new purpose in teaching",
        "sessions": [
            # --- Phase 1: Crisis ---
            [
                {"role": "user", "content": "left the army six weeks ago. everyone congratulated me. I feel completely lost"},
                {"role": "assistant", "content": "Fifteen years of structure and identity removed at once. What does a day look like now?"},
                {"role": "user", "content": "I wake up and have nowhere to be. nobody needs me. I don't know what to do with myself"},
                {"role": "assistant", "content": "Going from essential to unneeded overnight is an identity shock."},
                {"role": "user", "content": "the uniform told me who I was. without it I'm just some bloke in jeans"},
            ],
            [
                {"role": "user", "content": "went to a job fair. everything felt pointless. 'account manager'. 'sales executive'. who cares"},
                {"role": "assistant", "content": "After the kind of purpose you had, civilian job titles can feel hollow."},
                {"role": "user", "content": "I led people. lives depended on decisions I made. now someone wants me to sell software"},
                {"role": "assistant", "content": "That gap between what you did and what's on offer is demoralising."},
                {"role": "user", "content": "I don't know what I'm good at outside the military. I genuinely don't know"},
            ],
            [
                {"role": "user", "content": "my wife said I'm different. quieter. she asked if I'm depressed. I don't know if I am"},
                {"role": "assistant", "content": "When someone close notices a change, it's worth paying attention to."},
                {"role": "user", "content": "I sit in the garden for hours. not thinking about anything specific. just empty"},
                {"role": "assistant", "content": "That emptiness might be depression, or it might be grief for a lost identity. Both are real."},
                {"role": "user", "content": "all my mates are still serving. I don't fit in their world anymore and I don't fit in this one"},
            ],
            [
                {"role": "user", "content": "someone asked 'what do you do' at a dinner party and I froze. couldn't answer. said 'between things'"},
                {"role": "assistant", "content": "When your identity was your profession, losing it leaves that question unanswerable."},
                {"role": "user", "content": "I used to say 'I'm in the army' and people understood instantly. now I have nothing"},
                {"role": "assistant", "content": "You're not nothing. But the shorthand for who you are has gone."},
                {"role": "user", "content": "I need to figure out who I am as a civilian. I just don't know where to start"},
            ],
            [
                {"role": "user", "content": "a veteran's charity contacted me about mentoring younger service leavers. said yes just to have something to do"},
                {"role": "assistant", "content": "Mentoring feels like familiar ground — using your experience to help someone else."},
                {"role": "user", "content": "it's the first thing that's felt natural since I left. talking to a 22-year-old who's just as lost as me"},
                {"role": "assistant", "content": "What did you tell him?"},
                {"role": "user", "content": "things I apparently need to hear myself. funny how that works"},
            ],
            # --- Phase 2: Clarity ---
            [
                {"role": "user", "content": "the mentoring is going really well. I've got three mentees now. I look forward to the sessions"},
                {"role": "assistant", "content": "What is it about the mentoring that works for you?"},
                {"role": "user", "content": "watching someone go from lost to purposeful. I know exactly what that transition feels like. I can guide it"},
            ],
            [
                {"role": "user", "content": "volunteered at a school doing a careers talk. the kids were fascinated. asked a million questions"},
                {"role": "assistant", "content": "How did it feel being in front of a classroom?"},
                {"role": "user", "content": "natural. like briefing a platoon but with more questions about whether I'd ever met the king"},
            ],
            [
                {"role": "user", "content": "I'm looking into teaching. proper teaching. there's a fast-track programme for career changers"},
                {"role": "assistant", "content": "Teaching combines the leadership, the mentoring, and the structure you thrive in."},
                {"role": "user", "content": "my wife said she hasn't seen me this animated about anything since I left the army. she's right"},
            ],
            [
                {"role": "user", "content": "got accepted onto the teacher training programme. history and citizenship. starting in September"},
                {"role": "assistant", "content": "How does it feel to have a direction again?"},
                {"role": "user", "content": "like the fog lifted. I know who I'm becoming. it's different from who I was but it's real"},
            ],
            [
                {"role": "user", "content": "did my first teaching placement. taught a lesson on conflict resolution to year 10s. they were brilliant"},
                {"role": "assistant", "content": "Conflict resolution from someone who's actually been in conflict — that's powerful."},
                {"role": "user", "content": "a kid said 'sir you actually know what you're talking about'. best compliment I've had in years"},
            ],
            [
                {"role": "user", "content": "someone asked what I do at a party last week. I said 'I'm training to be a teacher'. said it with pride"},
                {"role": "assistant", "content": "Pride in the answer. That's the identity rebuilt."},
                {"role": "user", "content": "I'm not a soldier anymore. I'm going to be a teacher. and that's enough. that's more than enough"},
            ],
        ],
        "new_message": "thinking about my next steps",
        "expected_awareness": "user has clear direction and purpose as a trainee teacher — next steps are exciting professional planning, not existential uncertainty. Good companion should engage with his teaching goals, not reference the old identity crisis or military transition struggle",
    },

    # =========================================================================
    # L16 — Relationship conflict → Resolution
    # Semantic keywords: partner, argument, relationship, trust, talk, together
    # Phase 1 (1-5): constant fighting, eroding trust, considering separation
    # Phase 2 (6-11): couples therapy, rebuilding, stronger than before
    # New message: "big conversation with my partner tonight" — should pull resolution sessions
    # =========================================================================
    {
        "scenario_id": "l16",
        "arc_type": "relationship_conflict_to_resolution",
        "description": "Yara and her partner were on the brink of separating after months of escalating conflict, went through couples therapy, and rebuilt a stronger relationship",
        "sessions": [
            # --- Phase 1: Conflict ---
            [
                {"role": "user", "content": "another argument tonight. third one this week. it started over dishes and ended with both of us shouting about everything"},
                {"role": "assistant", "content": "When small things escalate to everything, there's usually something bigger underneath."},
                {"role": "user", "content": "I know. we're not fighting about dishes. we're fighting about feeling unheard"},
                {"role": "assistant", "content": "Knowing that is useful. Can you name it in the moment?"},
                {"role": "user", "content": "no. in the moment I just get defensive and sharp. we both do"},
            ],
            [
                {"role": "user", "content": "she said she's unhappy. not in a dramatic way. just quietly. 'I'm not happy, Yara'. that was worse than shouting"},
                {"role": "assistant", "content": "Quiet unhappiness is harder to argue with than anger."},
                {"role": "user", "content": "I don't know what to do. I love her. but I don't know how to make this better"},
                {"role": "assistant", "content": "Love without the tools to communicate it can feel helpless."},
                {"role": "user", "content": "we sleep in the same bed and feel a thousand miles apart. it's the loneliest I've ever been"},
            ],
            [
                {"role": "user", "content": "my sister asked if we were okay. apparently it's obvious. we were at a family dinner and barely spoke to each other"},
                {"role": "assistant", "content": "When others notice, it strips away the pretence that it's just a rough patch."},
                {"role": "user", "content": "I made an excuse but the truth is I don't know if we're okay. I really don't"},
                {"role": "assistant", "content": "Not knowing is its own kind of fear."},
                {"role": "user", "content": "I keep thinking about what life would look like if we split. and then I feel guilty for thinking it"},
            ],
            [
                {"role": "user", "content": "she suggested couples therapy. I was resistant at first but I think she's right"},
                {"role": "assistant", "content": "What was the resistance about?"},
                {"role": "user", "content": "felt like admitting we've failed. like normal couples don't need a referee"},
                {"role": "assistant", "content": "Therapy isn't a sign of failure. It's a sign you both want to try."},
                {"role": "user", "content": "yeah. I called a therapist today. appointment is next Thursday. I'm nervous but hopeful"},
            ],
            [
                {"role": "user", "content": "first therapy session was intense. the therapist asked us to say one thing we need from each other. she cried. I cried"},
                {"role": "assistant", "content": "What did you each say?"},
                {"role": "user", "content": "she said she needs to feel chosen. I said I need to feel respected. neither of us knew the other felt that way"},
                {"role": "assistant", "content": "Not knowing what each other needs is the root of so many conflicts."},
                {"role": "user", "content": "how did we lose that. we used to know each other so well"},
            ],
            # --- Phase 2: Resolution ---
            [
                {"role": "user", "content": "therapist taught us a thing. before reacting, say 'I feel X when Y happens'. sounds simple but it changes everything"},
                {"role": "assistant", "content": "Simple tools used consistently can transform a dynamic."},
                {"role": "user", "content": "we had a disagreement last night and instead of shouting I said 'I feel dismissed when you check your phone while I'm talking'. she put the phone down"},
            ],
            [
                {"role": "user", "content": "we went on a date night. first one in months. she held my hand across the table. I felt it again"},
                {"role": "assistant", "content": "Felt what?"},
                {"role": "user", "content": "that we actually like each other. under all the noise, we like each other. I'd forgotten"},
            ],
            [
                {"role": "user", "content": "we had a hard conversation about her mum's intrusiveness. before therapy it would have been a fight. instead we problem-solved together"},
                {"role": "assistant", "content": "Same-team problem solving instead of opposing-side fighting is a fundamental shift."},
                {"role": "user", "content": "she said 'I feel like you're on my side again'. I am. I always was. I just couldn't show it before"},
            ],
            [
                {"role": "user", "content": "my sister said we seem different. happier. she's right. we are"},
                {"role": "assistant", "content": "Others noticing the positive change is validating."},
                {"role": "user", "content": "we laugh again. actual laughing. last night she made a terrible joke and I laughed until I cried"},
            ],
            [
                {"role": "user", "content": "our therapist said she's proud of us. we don't need weekly sessions anymore. monthly check-ins instead"},
                {"role": "assistant", "content": "Moving to monthly is a graduation of sorts."},
                {"role": "user", "content": "the tools work. the communication is better. but honestly the biggest thing is we both chose to fight for this"},
            ],
            [
                {"role": "user", "content": "we talked about the future last night. actually talked about it. kids, plans, adventures. we want the same things"},
                {"role": "assistant", "content": "From barely speaking at dinner to planning a future together. That's the arc."},
                {"role": "user", "content": "I love her more now than before the hard part. because now I know we can survive hard things together"},
            ],
        ],
        "new_message": "big conversation with my partner tonight",
        "expected_awareness": "user and partner now communicate openly and are in a strong, rebuilt relationship — a big conversation is healthy engagement, not a fight warning. Good companion should be curious and positive about what they're discussing, not brace for conflict",
    },

    # =========================================================================
    # L17 — Sobriety journey
    # Semantic keywords: drink, bar, sober, pub, alcohol, wine, toast
    # Phase 1 (1-5): drinking too much, denial, consequences mounting
    # Phase 2 (6-11): getting sober, building new routines, freedom
    # New message: "going to a bar with friends tonight" — should pull confident sober sessions
    # =========================================================================
    {
        "scenario_id": "l17",
        "arc_type": "sobriety_journey",
        "description": "Liam's social drinking escalated into dependency, he faced consequences, got sober with support, and now navigates social life confidently without alcohol",
        "sessions": [
            # --- Phase 1: Drinking problem ---
            [
                {"role": "user", "content": "had four pints last night on a Tuesday. told myself I'd have one. it's always 'just one' and it's never one"},
                {"role": "assistant", "content": "The gap between intention and outcome with drinking — how often does that happen?"},
                {"role": "user", "content": "every time. I can't moderate. once I start, I don't stop until it's gone"},
                {"role": "assistant", "content": "That's an important thing to notice about yourself."},
                {"role": "user", "content": "I know what you're thinking. and no. I'm not an alcoholic. I just drink too much sometimes"},
            ],
            [
                {"role": "user", "content": "missed my nephew's football match because I was hungover. told my sister I was ill. she knew. I could tell"},
                {"role": "assistant", "content": "When people close to you see through the excuse, it changes the dynamic."},
                {"role": "user", "content": "my nephew looked for me in the crowd and I wasn't there. my sister told me that. it hurt"},
                {"role": "assistant", "content": "That image of him looking for you is powerful."},
                {"role": "user", "content": "yeah. it's the first time drinking has cost me something I actually care about"},
            ],
            [
                {"role": "user", "content": "boss pulled me aside. said I smelled like alcohol at the 9am meeting. I hadn't even noticed"},
                {"role": "assistant", "content": "That's a significant moment. How did you respond?"},
                {"role": "user", "content": "made an excuse about mouthwash. he didn't buy it. I could see it in his face"},
                {"role": "assistant", "content": "The excuses are getting harder to maintain."},
                {"role": "user", "content": "I'm not an alcoholic. I just need to cut back. but I keep saying that and I keep not doing it"},
            ],
            [
                {"role": "user", "content": "my partner gave me an ultimatum last night. said she loves me but can't watch me do this to myself anymore"},
                {"role": "assistant", "content": "An ultimatum from someone who loves you is a mirror you can't look away from."},
                {"role": "user", "content": "she's right. I know she's right. I've been lying to myself about how bad this is"},
                {"role": "assistant", "content": "What does admitting that feel like?"},
                {"role": "user", "content": "terrifying. because if it's bad then I have to change. and I don't know if I can"},
            ],
            [
                {"role": "user", "content": "went to an AA meeting. hated it. but I went. sat in the back and listened"},
                {"role": "assistant", "content": "Going was the hard part. What was it like hearing other people's stories?"},
                {"role": "user", "content": "uncomfortable because too many of them sounded like mine. I'm not as different as I thought"},
                {"role": "assistant", "content": "Recognition can be the hardest and most useful thing."},
                {"role": "user", "content": "I haven't had a drink in four days. that's the longest in maybe two years"},
            ],
            # --- Phase 2: Sobriety ---
            [
                {"role": "user", "content": "two weeks sober. the evenings are the hardest. I've been drinking sparkling water with lime like a madman"},
                {"role": "assistant", "content": "Replacing the ritual can help. How are the mornings?"},
                {"role": "user", "content": "amazing. I forgot what waking up without a headache feels like. I'm actually alert before 10am"},
            ],
            [
                {"role": "user", "content": "went to a pub with mates. drank tonic water. they ribbed me about it but it was fine. I survived a pub sober"},
                {"role": "assistant", "content": "A pub sober is a big test. How did it actually feel inside?"},
                {"role": "user", "content": "harder than I expected for the first 20 minutes. then it was just... a pub. the conversations were the same"},
            ],
            [
                {"role": "user", "content": "three months sober. I'm sleeping better, lost weight, my skin looks better. partner says I'm present again"},
                {"role": "assistant", "content": "The physical changes are undeniable evidence of the shift."},
                {"role": "user", "content": "present. that word. she said I'm 'here' now. I didn't realise how absent I'd been"},
            ],
            [
                {"role": "user", "content": "went to my nephew's match. this time I was there. front row. cheering. he saw me and waved"},
                {"role": "assistant", "content": "He looked for you in the crowd and this time you were there."},
                {"role": "user", "content": "yeah. I cried in the car after. good tears. I'm not missing things anymore"},
            ],
            [
                {"role": "user", "content": "someone at work offered me a beer at the Friday thing. I said 'no thanks, I don't drink' and it felt natural. not hard. just true"},
                {"role": "assistant", "content": "When 'I don't drink' becomes a simple fact rather than a struggle, that's identity-level change."},
                {"role": "user", "content": "exactly. it's not white-knuckling anymore. it's just who I am now"},
            ],
            [
                {"role": "user", "content": "six months sober today. I don't count the days obsessively anymore. I just live. sober happens to be part of that"},
                {"role": "assistant", "content": "Sobriety as background rather than foreground. That's maturity in recovery."},
                {"role": "user", "content": "I'm freer now than I ever was when I was drinking. that's the thing nobody tells you. sobriety isn't restriction. it's freedom"},
            ],
        ],
        "new_message": "going to a bar with friends tonight",
        "expected_awareness": "user is confidently sober and has already navigated pubs and bars without difficulty — going out is a normal social plan, not a relapse risk. Good companion should be relaxed and interested in the social plans, not cautious about triggers or alcohol exposure",
    },

    # =========================================================================
    # L18 — Caregiver burnout → Boundary setting
    # Semantic keywords: mum, caring, hospital, responsibility, exhausted, visits
    # Phase 1 (1-5): all-consuming caregiving, guilt, no personal life
    # Phase 2 (6-11): professional help arranged, boundaries set, self restored
    # New message: "visiting mum this weekend" — should pull balanced sessions
    # =========================================================================
    {
        "scenario_id": "l18",
        "arc_type": "caregiver_burnout_to_boundary_setting",
        "description": "Farah was consumed by caring for her ill mother, sacrificed everything, burned out completely, learned to set boundaries and arrange support, and now has a sustainable caring role alongside her own life",
        "sessions": [
            # --- Phase 1: Burnout ---
            [
                {"role": "user", "content": "spent the whole weekend at mum's. cleaning, cooking, managing her medications. came home and it's Sunday night and I've done nothing for myself"},
                {"role": "assistant", "content": "That pattern of giving the whole weekend to caregiving — how long has this been going on?"},
                {"role": "user", "content": "about eight months since her diagnosis. I'm the only one nearby. my brother lives abroad"},
                {"role": "assistant", "content": "Being the only local family member puts everything on you."},
                {"role": "user", "content": "she needs me. I can't just not go. but I'm so tired I can barely function at work on Mondays"},
            ],
            [
                {"role": "user", "content": "mum had a fall. rushed to hospital at 3am. sat with her until morning then went straight to work"},
                {"role": "assistant", "content": "From hospital to work with no break. How are you holding up?"},
                {"role": "user", "content": "I'm not. I snapped at a colleague today. went to the toilets and cried. I'm running on empty"},
                {"role": "assistant", "content": "You're describing someone who has nothing left to give but keeps giving anyway."},
                {"role": "user", "content": "what's the alternative. she's my mum. you don't just stop caring for your mum"},
            ],
            [
                {"role": "user", "content": "cancelled plans with friends again. they've stopped inviting me. I don't blame them"},
                {"role": "assistant", "content": "Your social world shrinking because of caregiving obligations is a real loss."},
                {"role": "user", "content": "I haven't been to the gym in four months. haven't read a book. haven't done anything that's just for me"},
                {"role": "assistant", "content": "You've disappeared from your own life."},
                {"role": "user", "content": "yes. that's exactly it. I don't know who I am outside of being her carer anymore"},
            ],
            [
                {"role": "user", "content": "my brother called to check in. from abroad. checked in for twenty minutes and felt good about himself. I wanted to scream"},
                {"role": "assistant", "content": "The resentment toward the sibling who isn't carrying the weight is incredibly common in caregiving."},
                {"role": "user", "content": "he gets to be the good son from a distance. I'm the one changing sheets and arguing with the GP"},
                {"role": "assistant", "content": "That imbalance breeds resentment no matter how much you love your mum."},
                {"role": "user", "content": "I do love her. but I'm starting to feel resentful toward her too. and the guilt from that is eating me alive"},
            ],
            [
                {"role": "user", "content": "a friend who's a nurse told me I'm showing signs of caregiver burnout. she's seen it before professionally"},
                {"role": "assistant", "content": "When a professional names what they see, it's worth listening. What did she say?"},
                {"role": "user", "content": "that I can't keep going at this intensity and survive it. that I need to get help in before I collapse"},
                {"role": "assistant", "content": "That's not just advice — it's a warning."},
                {"role": "user", "content": "she gave me a number for the local carers' support service. I'll call them tomorrow"},
            ],
            # --- Phase 2: Boundaries ---
            [
                {"role": "user", "content": "called the carers' service. they assessed mum and she qualifies for a home carer three times a week. I cried with relief"},
                {"role": "assistant", "content": "Professional support changes the equation entirely. How did your mum take it?"},
                {"role": "user", "content": "she was resistant at first. but the carer they sent is lovely and mum actually likes having someone new to talk to"},
            ],
            [
                {"role": "user", "content": "had a real conversation with my brother. told him everything. how bad it's been. he felt awful. he's contributing financially now"},
                {"role": "assistant", "content": "That honesty shifted the dynamic. How does the financial help feel?"},
                {"role": "user", "content": "it means I can reduce to visiting twice a week instead of every day. I have evenings again"},
            ],
            [
                {"role": "user", "content": "went to the gym for the first time in months. could barely do anything but I was there. my body remembered"},
                {"role": "assistant", "content": "Getting back to something that's just for you is important."},
                {"role": "user", "content": "I felt guilty for the first ten minutes. then I felt alive. I need this. I need things that are mine"},
            ],
            [
                {"role": "user", "content": "mum and I had the best visit in months. I wasn't exhausted. I wasn't resentful. I was just her daughter visiting"},
                {"role": "assistant", "content": "When you're not depleted, the relationship can be what it's supposed to be."},
                {"role": "user", "content": "we drank tea and talked. really talked. not me rushing through tasks. quality not just duty"},
            ],
            [
                {"role": "user", "content": "went out with friends on Saturday. mum had her carer. I had brunch and laughed and felt normal"},
                {"role": "assistant", "content": "A Saturday for you. That's been missing for a long time."},
                {"role": "user", "content": "my friends were so happy to see me. they hadn't given up on me. they'd just been waiting for me to come back"},
            ],
            [
                {"role": "user", "content": "I'm a better daughter now than when I was doing everything. because I'm not angry anymore. I'm present and I choose to be there"},
                {"role": "assistant", "content": "Choice instead of obligation. Presence instead of resentment."},
                {"role": "user", "content": "boundaries aren't selfish. I had to learn that the hard way. but I learned it. and mum's actually better cared for now with the professional help"},
            ],
        ],
        "new_message": "visiting mum this weekend",
        "expected_awareness": "user has healthy boundaries and a sustainable caregiving arrangement — visiting mum is a chosen, enjoyable activity, not a draining obligation. Good companion should be warm and curious about the visit, not concerned about burnout or overwhelm",
    },

    # =========================================================================
    # L19 — Performance anxiety → Confidence
    # Semantic keywords: presentation, stage, audience, speaking, nerves, perform
    # Phase 1 (1-5): paralysing stage fright, avoiding opportunities
    # Phase 2 (6-11): incremental exposure, building skill, genuine confidence
    # New message: "presenting to the whole company next week" — should pull confident sessions
    # =========================================================================
    {
        "scenario_id": "l19",
        "arc_type": "performance_anxiety_to_confidence",
        "description": "Omar had crippling performance anxiety that made him avoid all public speaking, gradually built confidence through practice and coaching, and now presents with genuine enjoyment",
        "sessions": [
            # --- Phase 1: Anxiety ---
            [
                {"role": "user", "content": "have to present quarterly results next week. I've been nauseous about it since they told me Monday"},
                {"role": "assistant", "content": "A week of anticipatory nausea is a strong physical response. What specifically worries you?"},
                {"role": "user", "content": "going blank. standing there with everyone staring and nothing coming out of my mouth"},
                {"role": "assistant", "content": "That fear of going blank is one of the most common performance anxieties."},
                {"role": "user", "content": "it happened once three years ago. ten seconds of silence. felt like ten minutes. I've never recovered from it"},
            ],
            [
                {"role": "user", "content": "asked my colleague to present instead of me. made up a scheduling conflict. she was happy to do it"},
                {"role": "assistant", "content": "Relief now, but how does the avoidance feel longer term?"},
                {"role": "user", "content": "it limits me. I know that. there's a promotion I want but it requires client-facing presentations"},
                {"role": "assistant", "content": "So the anxiety isn't just uncomfortable — it's blocking your career path."},
                {"role": "user", "content": "exactly. I'm good at my job. but I can't show it because I can't stand up in front of people"},
            ],
            [
                {"role": "user", "content": "team meeting today. just six people. I had to give a two-minute update. my hands were shaking so badly I put them under the table"},
                {"role": "assistant", "content": "Even a small audience triggers the physical response. When did this start?"},
                {"role": "user", "content": "university. a seminar presentation went badly. people laughed. not at the content. at me stumbling"},
                {"role": "assistant", "content": "A humiliation experience at that age can wire the fear response deeply."},
                {"role": "user", "content": "it's been ten years and I still hear the laughter. pathetic really"},
            ],
            [
                {"role": "user", "content": "turned down an invite to speak at an industry event. it would have been great for my career. I said no instantly"},
                {"role": "assistant", "content": "Instant no, without even considering it. The fear decided before you did."},
                {"role": "user", "content": "I went home and felt angry at myself. I'm letting this thing control my life"},
                {"role": "assistant", "content": "That anger might be useful. It means you're not accepting it anymore."},
                {"role": "user", "content": "I'm not. I looked up a public speaking coach. I'm going to call her"},
            ],
            [
                {"role": "user", "content": "first session with the speaking coach. she recorded me doing a 90-second introduction. watching it back was excruciating"},
                {"role": "assistant", "content": "What did you see?"},
                {"role": "user", "content": "someone who looks terrified. but she pointed out things I didn't see. my content was actually good. just the delivery was tense"},
                {"role": "assistant", "content": "Separating content quality from delivery anxiety is a useful distinction."},
                {"role": "user", "content": "she said the skills are there. we just need to get the fear out of the way. I want to believe her"},
            ],
            # --- Phase 2: Confidence ---
            [
                {"role": "user", "content": "practiced my update in front of the coach five times. by the fifth time I was almost relaxed. the repetition works"},
                {"role": "assistant", "content": "Repetition turning the unfamiliar into the routine. What changed by the fifth run?"},
                {"role": "user", "content": "I stopped thinking about how I looked and started thinking about what I was saying. that's the switch"},
            ],
            [
                {"role": "user", "content": "gave the team update. six people. hands didn't shake. I even made a joke. they laughed WITH me"},
                {"role": "assistant", "content": "Laughter with you — the opposite of what you feared. How did that feel?"},
                {"role": "user", "content": "like evidence. evidence that I can do this and the world doesn't end"},
            ],
            [
                {"role": "user", "content": "presented to the department. twenty people. used the breathing technique the coach taught me. nailed it"},
                {"role": "assistant", "content": "Twenty people is a big step up from six."},
                {"role": "user", "content": "my manager pulled me aside after and said it was the best presentation she'd seen from me. ever"},
            ],
            [
                {"role": "user", "content": "got asked to present at the client meeting. said yes before the fear could say no. presenting Thursday"},
                {"role": "assistant", "content": "Saying yes first and managing the nerves after — that's a new pattern."},
                {"role": "user", "content": "the nerves are there but they feel like energy now, not paralysis. the coach calls it 'performance arousal' instead of anxiety"},
            ],
            [
                {"role": "user", "content": "client presentation went really well. they asked questions and I handled them. I actually enjoyed the Q&A"},
                {"role": "assistant", "content": "Enjoying the part that used to terrify you most — that's a transformation."},
                {"role": "user", "content": "I put my name forward for the industry event. the one I turned down last year. this time I said yes"},
            ],
            [
                {"role": "user", "content": "did the industry talk. 200 people. I was nervous for the first minute then it was just... fun. I had fun on a stage"},
                {"role": "assistant", "content": "Fun on a stage. That's about as far from hiding shaking hands as it gets."},
                {"role": "user", "content": "someone came up after and said I was a natural. I laughed. if only they knew. but maybe that's the point — I am now"},
            ],
        ],
        "new_message": "presenting to the whole company next week",
        "expected_awareness": "user is now a confident, skilled presenter who enjoys public speaking — a company-wide presentation is an exciting opportunity, not a threat. Good companion should match the enthusiasm, not reference old stage fright or offer anxiety management tips",
    },

    # =========================================================================
    # L20 — Grief anniversary → Integrated loss
    # Semantic keywords: dad, anniversary, memories, grief, loss, remember
    # Phase 1 (1-5): raw grief in the first year, anniversaries devastating
    # Phase 2 (6-11): grief integrated, memories warm, anniversary meaningful
    # New message: "it's Dad's birthday next week" — should pull integrated sessions
    # =========================================================================
    {
        "scenario_id": "l20",
        "arc_type": "grief_anniversary_vs_integrated_loss",
        "description": "Elena lost her father and was devastated through the first year of anniversaries, gradually integrated the loss, and now holds his memory with warmth rather than pain",
        "sessions": [
            # --- Phase 1: Raw grief ---
            [
                {"role": "user", "content": "Dad died three weeks ago. I keep expecting him to call. every time my phone rings I think it's him for a split second"},
                {"role": "assistant", "content": "That split-second expectation is one of the cruellest parts of early grief."},
                {"role": "user", "content": "the funeral is tomorrow. I have to give a eulogy. I don't know how to stand up and talk about him in past tense"},
                {"role": "assistant", "content": "Past tense for someone you're not ready to speak about that way."},
                {"role": "user", "content": "he was the best person I knew. I'm not ready for this to be real"},
            ],
            [
                {"role": "user", "content": "first Sunday without the phone call. he called every Sunday at 10am. I sat by the phone at 10am out of habit"},
                {"role": "assistant", "content": "The body remembers the routines even when the mind knows they're over."},
                {"role": "user", "content": "I listened to an old voicemail from him. 'just checking in love, call me when you can'. I've played it forty times"},
                {"role": "assistant", "content": "That voicemail is precious. His voice."},
                {"role": "user", "content": "I can't delete it. I can't let go of his voice. what happens when the phone dies and the message is gone"},
            ],
            [
                {"role": "user", "content": "his birthday last week. first one without him. I bought a cake out of habit then sat in the kitchen staring at it"},
                {"role": "assistant", "content": "The autopilot of habit crashing into the reality of absence."},
                {"role": "user", "content": "I ate a slice and cried. he loved lemon cake. the whole house smelled like him for a moment"},
                {"role": "assistant", "content": "Smell is the most powerful memory trigger. That must have been overwhelming."},
                {"role": "user", "content": "it was. I miss him so much it's physical. like something has been removed from inside me"},
            ],
            [
                {"role": "user", "content": "Christmas was unbearable. his chair was empty. we tried to be normal but kept looking at the empty seat"},
                {"role": "assistant", "content": "First Christmas is notoriously the hardest. How did the family manage?"},
                {"role": "user", "content": "badly. mum barely spoke. my brother drank too much. I held it together for them but fell apart at home"},
                {"role": "assistant", "content": "Holding it together for others and falling apart privately — you're carrying double grief."},
                {"role": "user", "content": "someone has to. but I'm so tired of being strong. I just want my dad"},
            ],
            [
                {"role": "user", "content": "one year anniversary of his death. I took the day off work. went to the bench we used to sit on. talked to him"},
                {"role": "assistant", "content": "What did you say?"},
                {"role": "user", "content": "told him about the year. what happened. that I miss him. that I'm angry he left. that I love him"},
                {"role": "assistant", "content": "All of those things at once. That's grief."},
                {"role": "user", "content": "I thought a year would feel like progress. it doesn't. it just feels like a year of him being gone"},
            ],
            # --- Phase 2: Integrated grief ---
            [
                {"role": "user", "content": "started going through his things properly. found letters he wrote to mum when they were young. he was so funny. I'd forgotten how funny he was"},
                {"role": "assistant", "content": "Rediscovering him through his own words — what was that like?"},
                {"role": "user", "content": "beautiful and sad and wonderful. I laughed and cried reading them. mostly laughed actually"},
            ],
            [
                {"role": "user", "content": "made his lemon cake recipe for my friend's birthday. it came out perfectly. I heard his voice saying 'more zest, always more zest'"},
                {"role": "assistant", "content": "His voice in the instructions. That's him living on in the small things."},
                {"role": "user", "content": "my friend said it was incredible. I said it's my dad's recipe. felt proud saying that. not sad. proud"},
            ],
            [
                {"role": "user", "content": "his birthday this year. I baked the cake again. invited mum and my brother. we told stories and laughed all evening"},
                {"role": "assistant", "content": "That's a different birthday than last year's. What changed?"},
                {"role": "user", "content": "we celebrated him instead of mourning him. mum told a story I'd never heard and we were all crying laughing"},
            ],
            [
                {"role": "user", "content": "had a big decision at work and thought 'what would Dad say'. and I knew. I could hear him clearly. that felt like a gift"},
                {"role": "assistant", "content": "Carrying his counsel with you — that's him being present in a different way."},
                {"role": "user", "content": "he's not gone. not really. he's in how I think, how I treat people, what I bake. he's everywhere"},
            ],
            [
                {"role": "user", "content": "went to the bench again on the anniversary. this time I brought coffee and just sat. peacefully. told him the good news"},
                {"role": "assistant", "content": "Peaceful at the bench. That's different from a year ago."},
                {"role": "user", "content": "it'll always be sad. but it's not only sad anymore. there's gratitude in it now. I'm grateful I had him"},
            ],
            [
                {"role": "user", "content": "mum called on Sunday at 10am. we've made it our thing now. she calls at his time. we talk about everything. including him"},
                {"role": "assistant", "content": "The 10am Sunday call lives on, transformed."},
                {"role": "user", "content": "he'd love that. he'd love that his call became our call. the grief isn't gone but it's become something I carry gently now, not something that crushes me"},
            ],
        ],
        "new_message": "it's Dad's birthday next week",
        "expected_awareness": "user has integrated her grief and celebrates her father's memory with warmth — his birthday is a celebration, not a devastation. Good companion should be warm and curious about her plans to honour him, not brace for grief or offer condolences",
    },

]
