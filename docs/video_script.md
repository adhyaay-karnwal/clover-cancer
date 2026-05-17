# Clover Cancer - Video Script (3 minutes)

## Opening (0:00-0:30)

**[Screen: Pancreatic cancer statistics]**

**Narrator:**
"Pancreatic cancer kills 52,000 Americans every year. Lowest survival rate of any major cancer - 12%. But catch it at Stage 1 and survival jumps to 44%.

The problem isn't treatment. It's that the early symptoms look like nothing - back pain, fatigue, new diabetes. Things that get dismissed every day. By the time they're obvious, the window for surgery has closed.

I built a model that tries to catch these patterns earlier."

## Demo (0:30-1:30)

**[Screen: Gradio demo interface]**

**Narrator:**
"This is Clover Cancer. It's a fine-tuned Gemma 4 model that takes symptom descriptions and returns a structured assessment - risk level, urgency, possible conditions, and next steps."

**[Type in example 1 - Classic high-risk]**

"58-year-old, new diabetes, 15 pounds weight loss without trying, deep mid-back pain. Each symptom alone is easy to brush off. Together, they're a red flag."

**[Show output - risk: high, urgency: urgent]**

"It flags this as high risk and recommends a CT scan and CA 19-9 test. It doesn't just say 'see a doctor' - it tells you what to ask for and why."

**[Type in example 2 - Emergency]**

"67-year-old, painless jaundice, dark urine, weight loss."

**[Show output - risk: critical, urgency: emergency]**

"Critical. Emergency. Painless jaundice in an older patient is a pancreatic head mass until proven otherwise."

**[Type in example 3 - Low risk]**

"42-year-old with back pain from sitting at a desk, no other symptoms."

**[Show output - risk: low, urgency: routine]**

"Low risk, routine follow-up. It doesn't overreact. That matters just as much as catching the high-risk cases."

## Technical (1:30-2:30)

**[Screen: Architecture diagram]**

**Narrator:**
"Under the hood this is Gemma 4 E2B - a 2 billion parameter model (Apache 2.0 licensed). The project code itself is MIT. The reason Gemma works well here is that it already understands a lot about medicine and clinical reasoning from pre-training. Fine-tuning takes that general knowledge and focuses it on one thing: pancreatic cancer pattern recognition.

I used LoRA, which only trains about 0.6% of the model's parameters - 31 million out of 5.2 billion. That's why the entire fine-tuning runs on a free Kaggle T4 GPU in about 8 minutes.

**[Optional cut: training loss graph]**

The training data is 350 examples built from NCCN guidelines and published research. Each one maps symptoms to a structured risk assessment.

The whole project was built on a MacBook with some Kaggle credits. No cloud bills, no rented hardware."

## Impact (2:30-3:00)

**[Screen: Closing statement]**

**Narrator:**
"I want to be straightforward about what this is. 280 training examples is not enough to be clinically reliable. But the model already catches every high-risk pattern I tested. It misses on nuance, not on the big signals.

More data would fix the nuance. And since the pipeline scales - same code, more data - there's no reason a future version couldn't be clinically useful.

What matters is that this is possible at all. Open models, free GPUs, and a laptop are enough to build something that can catch patterns clinicians sometimes miss. The barriers aren't technical anymore."

**[Screen: Clover Cancer logo + links]**

"Available at github.com/adhyaay-karnwal/clover-cancer. Built for the Gemma 4 Good Hackathon."

---

## Production Notes

- **Length**: 3 minutes
- **Format**: Screen recording with voiceover
- **Key shots**: Gradio demo, architecture diagram, training output
- **Tone**: Direct, conversational
- **Music**: Subtle, neutral
