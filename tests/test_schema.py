import pytest
from pydantic import ValidationError

from app.schema import CallToolAction
from app.schema import DraftVerdictAction
from app.schema import parse_agent_action
from app.schema import validate_tool_arguments


def test_parses_valid_tool_action():
    payload = {
        "action": "CALL_TOOL",
        "tool": "fuzzy_score",
        "arguments": {
            "field_name": "account_name",
        },
        "reason": "The account names are not exactly equal.",
    }

    action = parse_agent_action(payload)
    arguments = validate_tool_arguments(action)

    assert isinstance(action, CallToolAction)
    assert arguments == {
        "field_name": "account_name",
    }


def test_rejects_invalid_tool_arguments():
    payload = {
        "action": "CALL_TOOL",
        "tool": "fuzzy_score",
        "arguments": {
            "field_name": "priority",
        },
        "reason": "Check priority similarity.",
    }

    action = parse_agent_action(payload)

    with pytest.raises(ValidationError):
        validate_tool_arguments(action)


def test_parses_valid_duplicate_verdict():
    payload = {
        "action": "DRAFT_VERDICT",
        "recommendation": {
            "verdict": "DUPLICATE",
            "confidence": 0.92,
            "summary": "The second case is a follow-up.",
            "evidence": [
                {
                    "evidence_id": "tool-1",
                    "claim": "The contact emails match.",
                }
            ],
            "uncertainties": [],
        },
    }

    action = parse_agent_action(payload)

    assert isinstance(action, DraftVerdictAction)
    assert action.recommendation.confidence == 0.92


def test_rejects_unknown_verdict():
    payload = {
        "action": "DRAFT_VERDICT",
        "recommendation": {
            "verdict": "PROBABLY_DUPLICATE",
            "confidence": 0.8,
            "summary": "The cases look similar.",
            "evidence": [],
            "uncertainties": [],
        },
    }

    with pytest.raises(ValidationError):
        parse_agent_action(payload)


def test_rejects_confidence_outside_range():
    payload = {
        "action": "DRAFT_VERDICT",
        "recommendation": {
            "verdict": "UNSURE",
            "confidence": 1.5,
            "summary": "Insufficient evidence.",
            "evidence": [],
            "uncertainties": [],
        },
    }

    with pytest.raises(ValidationError):
        parse_agent_action(payload)


def test_decided_verdict_requires_evidence():
    payload = {
        "action": "DRAFT_VERDICT",
        "recommendation": {
            "verdict": "NOT_DUPLICATE",
            "confidence": 0.9,
            "summary": "The accounts differ.",
            "evidence": [],
            "uncertainties": [],
        },
    }

    with pytest.raises(ValidationError):
        parse_agent_action(payload)


def test_unsure_verdict_may_have_no_evidence():
    payload = {
        "action": "DRAFT_VERDICT",
        "recommendation": {
            "verdict": "UNSURE",
            "confidence": 0.0,
            "summary": "The investigation could not complete.",
            "evidence": [],
            "uncertainties": [
                "No validated tool evidence was available."
            ],
        },
    }

    action = parse_agent_action(payload)

    assert isinstance(action, DraftVerdictAction)
    assert action.recommendation.verdict.value == "UNSURE"
