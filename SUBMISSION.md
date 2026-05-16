# Kaggle Hackathon Submission Guide

Step-by-step instructions for submitting Clover Cancer to the Gemma 4 Good Hackathon.

## Deadline

**May 18, 2026 at 11:59 PM UTC**

## What You Need

1. **Kaggle Writeup** — Blog-style report (max 1500 words)
2. **YouTube Video** — Demo video (max 3 minutes)
3. **Public GitHub Repo** — Already done: `github.com/adhyaay-karnwal/clover-cancer`
4. **Live Demo** — Gradio app (files in repo, or deploy to HuggingFace Spaces)
5. **Media Gallery** — Cover image + screenshots

---

## Step 1: Create the Kaggle Writeup

1. Go to [kaggle.com/competitions/gemma-4-good-hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
2. Click **"New Writeup"** button
3. Fill in:
   - **Title**: "Clover Cancer: Pancreatic Cancer Early Detection with Fine-tuned Gemma 4"
   - **Subtitle**: "A specialized triage system to catch pancreatic cancer patterns that get missed"
   - **Track**: Select **"Health & Sciences"** (primary) — you can also mention Unsloth and Safety & Trust in the text
4. Copy the content from `docs/writeup.md` into the writeup body
5. In the **"Attachments"** section under **"Project Links"**:
   - **Code Repository**: `https://github.com/adhyaay-karnwal/clover-cancer`
   - **Live Demo**: `https://github.com/adhyaay-karnwal/clover-cancer` (or HuggingFace Spaces URL if deployed)
   - **Model Weights**: `https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-lora`
   - **Training Data**: `https://www.kaggle.com/datasets/adhyaaykarnwal/clover-cancer-data`
   - **Training Notebook**: `https://www.kaggle.com/code/adhyaaykarnwal/clover-cancer-gemma-4-fine-tuning`
6. In the **"Media Gallery"**:
   - Upload a **cover image** (required) — screenshot of the demo or a project logo
   - Upload the YouTube video link
7. Click **"Save"** (you can edit and re-submit as many times before deadline)

---

## Step 2: Record the Video

### Script (3 minutes max)

**0:00 - 0:30 — The Problem**
> "Pancreatic cancer is the deadliest major cancer. 5-year survival is just 12%. But when caught at Stage 1, survival jumps to 44%. The symptoms — back pain, new diabetes, weight loss — are vague and routinely dismissed. What if we could catch the patterns that get missed?"

**0:30 - 2:00 — Live Demo**
- Show the Gradio interface
- Enter the high-risk example: "58-year-old male, new-onset diabetes, 15 pounds weight loss, back pain"
- Show the model's output: high risk, urgent, specific conditions, reasoning chain
- Enter a low-risk example: "42-year-old male, back pain from desk job, no other symptoms"
- Show the model correctly identifies low risk

**2:00 - 2:30 — Technical Deep-Dive**
> "This is a fine-tuned Gemma 4 E2B model trained using Unsloth on Kaggle's free T4 GPU. 31 million trainable parameters, 0.6% of the model. Trained on 280 examples of pancreatic cancer symptom patterns grounded in NCCN clinical guidelines."

**2:30 - 3:00 — Vision**
> "This tool democratizes clinical pattern recognition. Not everyone has access to a gastroenterologist who connects new diabetes and back pain to pancreatic cancer. Early detection saves lives."

### Recording Tips

- Use OBS Studio (free) or QuickTime Player (Mac) for screen recording
- Record at 1080p
- Speak clearly and at a moderate pace
- Show the actual demo running — don't fake it
- Keep it under 3 minutes

### Upload to YouTube

1. Go to [youtube.com](https://youtube.com)
2. Click **Create** → **Upload video**
3. Set visibility to **Unlisted** (so judges can view without login)
4. Copy the video URL — you'll need it for the writeup

---

## Step 3: Attach Video to Writeup

1. Go back to your Kaggle writeup
2. In the **Media Gallery** section, paste the YouTube URL
3. The video should embed automatically

---

## Step 4: Make the Repo Public

The repo is already at `github.com/adhyaay-karnwal/clover-cancer`. To make it public:

1. Go to the repo on GitHub
2. Click **Settings** → scroll to **Danger Zone**
3. Click **Change repository visibility** → **Make public**
4. Confirm

---

## Step 5: Submit

1. Go to your Kaggle writeup
2. Click the **"Submit"** button (top right)
3. You can un-submit and re-submit as many times before the deadline

---

## Checklist

- [ ] Kaggle writeup created and under 1500 words
- [ ] Track selected (Health & Sciences)
- [ ] YouTube video uploaded (unlisted, under 3 min)
- [ ] Video attached to writeup Media Gallery
- [ ] GitHub repo link attached to writeup
- [ ] Cover image uploaded to Media Gallery
- [ ] GitHub repo made public
- [ ] Writeup submitted on Kaggle

---

## Troubleshooting

### "I can't find the Submit button"
The Submit button only appears after you save the writeup. Click Save first, then Submit will appear in the top right.

### "My video won't embed"
Make sure the YouTube video is set to **Unlisted** (not Private). Private videos can't be embedded.

### "I need to edit after submitting"
You can un-submit, edit, and re-submit as many times before the deadline. Just click the Submit button again after editing.

### "What track should I select?"
Primary track: **Health & Sciences**. You can mention Unsloth and Safety & Trust in the writeup text — projects can win multiple tracks.
