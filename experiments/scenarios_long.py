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

]
