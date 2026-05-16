"""
Clover Cancer — Gradio Demo Application

Interactive web interface for pancreatic cancer symptom triage.
Users can describe symptoms and receive structured risk assessments
with reasoning chains and recommended actions.

Author: Adhyaay Karnwal
"""

import gradio as gr
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.inference import PancreaticCancerTriage, format_assessment, SYSTEM_PROMPT


# Initialize triage engine
triage = PancreaticCancerTriage()

# Example scenarios for judges to try
EXAMPLE_SCENARIOS = [
    [
        "I'm a 58-year-old male. I was just diagnosed with diabetes last month. "
        "I've lost about 15 pounds without trying. I have this deep ache in my "
        "mid-back that won't go away. This has been going on for about 3 months.",
        "58",
        "Male",
        ["New-onset diabetes"],
        "3 months"
    ],
    [
        "I'm a 67-year-old female. My skin turned yellow and the whites of my "
        "eyes look yellow too. I've lost about 10 pounds in the last month. My "
        "urine is really dark. I haven't had any pain really.",
        "67",
        "Female",
        ["None"],
        "2 weeks"
    ],
    [
        "I'm a 52-year-old female with a BRCA2 mutation. My mother died of "
        "pancreatic cancer. I've been having abdominal pain and fatigue for "
        "about 6 weeks. I've lost some weight too.",
        "52",
        "Female",
        ["BRCA2 mutation", "Family history of pancreatic cancer"],
        "6 weeks"
    ],
    [
        "I'm a 42-year-old male. I've had back pain and some stomach discomfort "
        "for about 2 weeks. I sit at a desk all day. No weight loss or other symptoms.",
        "42",
        "Male",
        ["None"],
        "2 weeks"
    ],
    [
        "I'm a 63-year-old female. I've had two blood clots in my legs in the "
        "past year. I've also been having abdominal pain and lost about 15 pounds. "
        "My doctor can't figure out why I keep getting clots.",
        "63",
        "Female",
        ["History of DVT"],
        "1 year"
    ],
]


def assess_patient(symptoms, age, gender, risk_factors_list, duration):
    """Run triage assessment from Gradio inputs."""
    # Build patient description
    parts = [f"I'm a {age}-year-old {gender.lower()}."]
    parts.append(symptoms)
    if duration:
        parts.append(f"This has been going on for about {duration}.")
    if risk_factors_list and "None" not in risk_factors_list:
        rf = ", ".join(risk_factors_list)
        parts.append(f"My medical history includes: {rf}.")

    description = " ".join(parts)

    # Run assessment
    assessment = triage.assess(description)

    # Format for display
    formatted = format_assessment(assessment)

    # Determine risk color
    risk = assessment.get("risk_assessment", "unknown").lower()
    risk_colors = {
        "low": "#22c55e",
        "medium": "#f59e0b",
        "high": "#ef4444",
        "critical": "#991b1b",
        "unknown": "#6b7280"
    }
    color = risk_colors.get(risk, "#6b7280")

    # Format JSON for display
    json_display = json.dumps(assessment, indent=2)

    return formatted, json_display, color


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Clover Cancer — Pancreatic Cancer Early Detection",
        theme=gr.themes.Soft(),
        css="""
        .risk-high { color: #ef4444; font-weight: bold; }
        .risk-critical { color: #991b1b; font-weight: bold; }
        """
    ) as app:
        gr.Markdown("""
        # Clover Cancer
        ### Pancreatic Cancer Early Detection AI

        **What this does:** Analyzes patient symptoms to assess pancreatic cancer risk,
        provide differential diagnoses, and recommend next steps.

        **Why it matters:** Pancreatic cancer has a 12% 5-year survival rate — but when
        caught at Stage 1, survival jumps to 44%. This tool helps identify high-risk
        patterns that might otherwise be dismissed.

        **Disclaimer:** This is an AI research tool for educational purposes.
        It is NOT a medical device. Always consult a healthcare provider.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Patient Information")

                symptoms = gr.Textbox(
                    label="Describe your symptoms",
                    placeholder="Describe what you're experiencing in your own words...",
                    lines=4,
                )

                with gr.Row():
                    age = gr.Textbox(label="Age", placeholder="55")
                    gender = gr.Radio(
                        choices=["Male", "Female"],
                        label="Sex",
                        value="Male"
                    )

                risk_factors = gr.CheckboxGroup(
                    choices=[
                        "None",
                        "Family history of pancreatic cancer",
                        "BRCA1/BRCA2 mutation",
                        "Lynch syndrome",
                        "Chronic pancreatitis",
                        "New-onset diabetes",
                        "Smoking history",
                        "Obesity",
                        "History of DVT",
                    ],
                    label="Risk Factors",
                    value=["None"]
                )

                duration = gr.Textbox(
                    label="How long have you had these symptoms?",
                    placeholder="e.g., 2 weeks, 3 months"
                )

                assess_btn = gr.Button(
                    "Assess Symptoms",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Assessment Results")

                risk_display = gr.Markdown(
                    value="*Enter symptoms and click 'Assess' to see results*"
                )

                assessment_text = gr.Textbox(
                    label="Detailed Assessment",
                    lines=15,
                    interactive=False,
                )

                with gr.Accordion("Raw JSON Output", open=False):
                    json_output = gr.Code(
                        label="Structured Output",
                        language="json",
                    )

        gr.Markdown("""
        ---
        ### Example Scenarios

        Try these pre-filled examples to see how the model assesses different presentations:
        """)

        gr.Examples(
            examples=EXAMPLE_SCENARIOS,
            inputs=[symptoms, age, gender, risk_factors, duration],
            label="Click an example to load it"
        )

        gr.Markdown("""
        ---
        ### About This Project

        **Clover Cancer** is a fine-tuned Gemma 4 model trained on pancreatic cancer
        symptom patterns, risk factors, and clinical guidelines. Built for the
        [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon).

        **Key features:**
        - Identifies high-risk symptom patterns (new-onset diabetes + weight loss + back pain)
        - Considers genetic risk factors (BRCA2, Lynch syndrome, family history)
        - Provides structured reasoning chains for explainability
        - Recommends specific diagnostic actions (CT, MRI, EUS, CA 19-9)

        **Author:** Adhyaay Karnwal | [GitHub](https://github.com/adhyaay-karnwal)
        """)

        # Wire up the assessment
        assess_btn.click(
            fn=assess_patient,
            inputs=[symptoms, age, gender, risk_factors, duration],
            outputs=[assessment_text, json_output, risk_display],
        )

    return app


if __name__ == "__main__":
    print("=" * 50)
    print("Clover Cancer — Pancreatic Cancer Triage Demo")
    print("=" * 50)

    # Try to load model with LoRA adapters
    model_path = os.environ.get("MODEL_PATH", "outputs/clover-cancer-lora")
    if os.path.exists(model_path):
        triage.load_model(model_path)
    else:
        print(f"No model found at {model_path}. Using mock assessment.")
        print("Set MODEL_PATH environment variable or train a model first.")

    # Launch
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
