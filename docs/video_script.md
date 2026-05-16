# Clover Cancer — Video Script (3 minutes)

## Opening (0:00-0:30)

**[Screen: Pancreatic cancer statistics]**

**Narrator:**
"Pancreatic cancer kills 52,000 Americans every year. It has the lowest survival rate of any major cancer — just 12%. But here's the thing: when caught early, at Stage 1, survival jumps to 44%.

The problem isn't that we can't treat it. The problem is that we can't catch it. The symptoms — back pain, fatigue, new diabetes — they're dismissed. By the time they're obvious, it's too late.

What if we could catch the patterns that get missed?"

## Demo (0:30-1:30)

**[Screen: Gradio demo interface]**

**Narrator:**
"Meet Clover Cancer — a fine-tuned Gemma 4 model trained to recognize pancreatic cancer symptom patterns.

Let me show you how it works."

**[Type in example 1 — Classic high-risk]**

"This is a 58-year-old with new diabetes, weight loss, and back pain. Three symptoms that could each be explained away individually. But together? They're a recognized early warning pattern for pancreatic cancer.

Watch what the model does."

**[Show output — risk: high, urgency: urgent]**

"It flags this as high risk. Urgent. And it tells you exactly why — citing the research connecting new-onset diabetes to pancreatic cancer, recommending specific imaging and blood tests.

This isn't a chatbot reciting facts. This is pattern recognition trained on clinical guidelines."

**[Type in example 2 — Emergency]**

"Now a 67-year-old with painless jaundice. Yellow skin, dark urine, weight loss."

**[Show output — risk: critical, urgency: emergency]**

"Critical. Emergency referral. Painless jaundice in an older patient — this is the hallmark presentation of a pancreatic head mass until proven otherwise."

**[Type in example 3 — Low risk]**

"And here's the flip side — a 42-year-old with back pain and no risk factors. The model says low risk, routine follow-up. It doesn't cry wolf."

## Technical (1:30-2:30)

**[Screen: Architecture diagram]**

**Narrator:**
"Clover Cancer is built on Google's Gemma 4 E2B — a 2 billion parameter open model. I fine-tuned it using LoRA — Low-Rank Adaptation — which trains just 0.6% of the model's parameters while teaching it an entirely new skill.

The training data is 350 structured examples built from NCCN clinical guidelines, peer-reviewed research from MD Anderson and Johns Hopkins, and real clinical presentation patterns.

Each training example maps patient symptoms to structured risk assessments with reasoning chains. The model doesn't just say 'high risk' — it explains why, cites the pattern, and recommends specific next steps.

The entire pipeline runs locally on a MacBook with 24GB of unified memory. No cloud, no API calls. Privacy-first."

## Impact (2:30-3:00)

**[Screen: Impact statement]**

**Narrator:**
"Every year, thousands of pancreatic cancer cases are missed because the symptoms are too vague. New diabetes. Back pain. Weight loss. Individually, they're nothing. Together, they're a cry for help that gets ignored.

Clover Cancer catches what gets missed. It's not a replacement for doctors — it's a second opinion that's always available, always thorough, and never dismissive.

When the right tools are accessible to everyone, the possibilities for positive change are truly endless."

**[Screen: Clover Cancer logo + links]**

"Built for the Gemma 4 Good Hackathon. Open source, Apache 2.0 licensed. Available at github.com/adhyaay-karnwal/clover-cancer."

---

## Production Notes

- **Length**: 3 minutes max
- **Format**: Screen recording with voiceover
- **Key shots**: Gradio demo (live interaction), architecture diagram, statistics
- **Tone**: Professional but accessible, not clinical
- **Music**: Subtle, hopeful background track
- **Text overlays**: Key statistics, risk levels, model output highlights
