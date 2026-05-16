"""
Clover Cancer — Basic Smoke Tests

Validates the inference pipeline works correctly without requiring
a trained model (uses mock mode).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.inference import PancreaticCancerTriage, format_assessment


def test_mock_assessment():
    """Test that mock assessment returns valid structure."""
    triage = PancreaticCancerTriage()

    # High-risk scenario
    result = triage.assess(
        "I'm a 58-year-old male. I was just diagnosed with diabetes. "
        "I've lost 15 pounds without trying. I have back pain."
    )

    assert "risk_assessment" in result, "Missing risk_assessment"
    assert "urgency" in result, "Missing urgency"
    assert result["risk_assessment"] in ["low", "medium", "high", "critical"], \
        f"Invalid risk: {result['risk_assessment']}"
    print(f"High-risk test: {result['risk_assessment']} — PASS")


def test_low_risk_scenario():
    """Test that low-risk scenario is assessed correctly."""
    triage = PancreaticCancerTriage()

    result = triage.assess(
        "I'm a 30-year-old female. I've had a headache for 2 days."
    )

    assert result["risk_assessment"] == "low", \
        f"Expected low risk, got {result['risk_assessment']}"
    print(f"Low-risk test: {result['risk_assessment']} — PASS")


def test_format_assessment():
    """Test that formatting works."""
    assessment = {
        "risk_assessment": "high",
        "conditions": [
            {"name": "Pancreatic cancer", "likelihood": "high", "reasoning": "Test reasoning"}
        ],
        "urgency": "urgent",
        "recommended_actions": ["Get CT scan"],
        "reasoning_chain": "Test reasoning chain"
    }

    formatted = format_assessment(assessment)
    assert "HIGH" in formatted, "Risk level not in output"
    assert "Pancreatic cancer" in formatted, "Condition not in output"
    print("Format test — PASS")


if __name__ == "__main__":
    test_mock_assessment()
    test_low_risk_scenario()
    test_format_assessment()
    print("\nAll tests passed!")
