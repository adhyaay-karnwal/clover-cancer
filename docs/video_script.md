# Clover Cancer — Video Script (3 minutes)

## Opening (0:00-0:30)

**[Screen: Pancreatic cancer statistics]**

**Narrator:**
"Pancreatic cancer kills 52,000 Americans every year. Lowest survival rate of any major cancer — 12%. But if you catch it at Stage 1, that jumps to 44%.

The problem isn't treatment. The problem is that the symptoms look like nothing. Back pain, fatigue, new diabetes — things that get dismissed every day. By the time they're obvious, it's too late.

I built something to try and catch those patterns earlier."

## Demo (0:30-1:30)

**[Screen: Gradio demo interface]**

**Narrator:**
"This is Clover Cancer. It's a Gemma 4 model I fine-tuned to recognize pancreatic cancer symptom patterns. Let me show you."

**[Type in example 1 — Classic high-risk]**

"Here's a 58-year-old with new diabetes, weight loss, and back pain. Each symptom alone is easy to brush off. Together, they're a red flag."

**[Show output — risk: high, urgency: urgent]**

"It flags this as high risk and recommends specific next steps — a CT scan, a CA 19-9 blood test. It doesn't just say 'see a doctor.' It tells you what to ask for and why."

**[Type in example 2 — Emergency]**

"Now a 67-year-old with painless jaundice. Yellow skin, dark urine, weight loss."

**[Show output — risk: critical, urgency: emergency]**

"Critical. Emergency. Painless jaundice in an older person is basically a pancreatic head mass until proven otherwise. The model knows this."

**[Type in example 3 — Low risk]**

"And here's the other side — a 42-year-old with back pain from sitting at a desk. No weight loss, no risk factors."

**[Show output — risk: low, urgency: routine]**

"It says low risk, routine follow-up. It doesn't overreact. That matters just as much."

## Technical (1:30-2:30)

**[Screen: Architecture diagram]**

**Narrator:**
"Under the hood, this is Google's Gemma 4 E2B — a 2 billion parameter model that's open source under Apache 2.0. What makes Gemma 4 great for a task like this is that it already knows a lot about the world. It understands medicine, biology, and clinical reasoning at a general level from pre-training. Fine-tuning takes that general knowledge and focuses it on one thing: catching pancreatic cancer patterns.

I used LoRA, which only trains about 0.6% of the model's parameters. That's 31 million out of 5.2 billion. It means the fine-tuning runs on a free Kaggle T4 GPU in about 8 minutes.

**[Optional cut: show training output scrolling]**

The training data is 350 examples built from NCCN clinical guidelines and research from MD Anderson and Johns Hopkins. Each one maps symptoms to a structured assessment with reasoning.

I built this with a MacBook and some Kaggle credits. That's it. No cloud bills, no rented GPUs, no enterprise budget."

## Impact (2:30-3:00)

**[Screen: Impact statement]**

**Narrator:**
"The honest truth is that 280 training examples is not enough to be clinically reliable. You'd need thousands of real cases — curated by oncologists, validated against patient outcomes — to get there. But what this shows is that the approach works. With 280 examples and 8 minutes of training, the model already catches every high-risk pattern I tested. It misses on nuance, not on the big signals.

More data would fix that. More data means the model learns the edge cases, the statistical rarity, the things that make pancreatic cancer so hard to catch. And since the approach scales — more data just means more training time, the same pipeline — there's no reason a future version couldn't be genuinely useful in clinical settings.

Gemma 4 is Apache 2.0. LoRA is open. Kaggle GPUs are free. The only barrier to making this better is having access to the right data. And that's a problem worth solving."

**[Screen: Clover Cancer logo + links]**

"Built for the Gemma 4 Good Hackathon. Open source, MIT licensed. Available at github.com/adhyaay-karnwal/clover-cancer."

---

## Production Notes

- **Length**: 3 minutes
- **Format**: Screen recording with voiceover
- **Key shots**: Gradio demo (live interaction), architecture diagram, training output scroll
- **Tone**: Direct, conversational, not salesy
- **Music**: Subtle, neutral background
- **Text overlays**: Key statistics, risk levels, model output highlights
