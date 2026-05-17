# Clover Cancer - Video Script (3 minutes)

## Opening (0:00-0:30)

**[Screen: Pancreatic cancer statistics]**

**Narrator:**
"Pancreatic cancer kills 52,000 Americans every year. It has the lowest survival rate of any major cancer at just 12%. But catch it at Stage 1 and survival jumps to 44%.

The problem is not treatment. The problem is that the early symptoms look like nothing. Back pain, fatigue, new diabetes. Things that get dismissed every day. By the time they are obvious, the window for surgery has closed.

I built a model that tries to catch these patterns earlier."

## Demo (0:30-1:30)

**[Screen: Gradio demo interface]**

**Narrator:**
"This is Clover Cancer. It is a fine-tuned Gemma 4 model that takes symptom descriptions and returns a structured assessment with risk level, urgency, possible conditions, and next steps."

**[Type in example 1 - Classic high-risk]**

"Here is a 58-year-old with new diabetes, 15 pounds of weight loss, and deep mid-back pain. Each symptom alone is easy to brush off. Together they are a red flag."

**[Show output - risk: high, urgency: urgent]**

"It flags this as high risk and recommends a CT scan and CA 19-9 test. Instead of just saying see a doctor, it tells you what to ask for and why."

**[Type in example 2 - Emergency]**

"Now a 67-year-old with painless jaundice, dark urine, and weight loss."

**[Show output - risk: critical, urgency: emergency]**

"Critical, emergency. Painless jaundice in an older patient is a pancreatic head mass until proven otherwise."

**[Type in example 3 - Low risk]**

"And here is a 42-year-old with back pain from sitting at a desk, no other symptoms."

**[Show output - risk: low, urgency: routine]**

"Low risk, routine follow-up. It does not overreact, which matters just as much as catching the high-risk cases."

## Technical (1:30-2:30)

**[Screen: Architecture diagram]**

**Narrator:**
"Under the hood this is Gemma 4 E2B, a 2 billion parameter model that is Apache 2.0 licensed. My project code is MIT. The reason Gemma works well here is that it already understands a lot about medicine and clinical reasoning from pre-training. Fine-tuning takes that general knowledge and focuses it on one specific thing, which is pancreatic cancer pattern recognition.

I used LoRA, which only trains about 0.6% of the total parameters, or 31 million out of 5.2 billion including embeddings. That is why the entire fine-tuning runs on a free Kaggle T4 GPU in about 8 minutes.

**[Optional cut: training loss graph]**

The training data is 350 examples built from NCCN guidelines and published research, with each one mapping symptoms to a structured risk assessment.

The whole project was built on a MacBook with some Kaggle credits. No cloud bills or rented hardware."

## Impact (2:30-3:00)

**[Screen: Closing statement]**

**Narrator:**
"I want to be straightforward about what this is. 280 training examples is not enough to be clinically reliable. But the model already catches every high-risk pattern I tested. It misses on nuance, not on the big signals.

More data would fix the nuance, and since the pipeline scales with the same code and more data, there is no reason a future version could not be clinically useful.

What matters is that this is possible at all. Open models, free GPUs, and a laptop are enough to build something that can catch patterns clinicians sometimes miss. The barriers are not technical anymore."

**[Screen: Clover Cancer logo + links]**

"Available at github.com/adhyaay-karnwal/clover-cancer. Built for the Gemma 4 Good Hackathon."

---

## Production Notes

- **Length**: 3 minutes
- **Format**: Screen recording with voiceover
- **Key shots**: Gradio demo, architecture diagram, training output
- **Tone**: Direct, conversational
- **Music**: Subtle, neutral
