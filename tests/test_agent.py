from app.agent import run_investigation
from app.candidates import CandidatePair
from app.candidates import CandidateReason
from app.data import index_cases
from app.data import load_cases
from app.llm_client import ModelAttempt
from app.llm_client import ModelResult
from app.schema import CallToolAction
from app.schema import DraftVerdict
from app.schema import DraftVerdictAction
from app.schema import ToolName
from app.schema import VerdictValue


class ScriptedLlmClient:
    def __init__(self, actions):
        self.actions = list(actions)
        self.states = []
        self.model = "scripted-test-model"
        self.temperature = 0.1

    def request_action(self, state):
        self.states.append(state)
        action = self.actions.pop(0)

        attempt = ModelAttempt(
            attempt_number=1,
            status="success",
            raw_response=action.model_dump_json(),
            error=None,
        )

        return ModelResult(
            action=action,
            attempts=[attempt],
        )


def load_investigation_data():
    cases = load_cases()
    case_index = index_cases(cases)
    pair = CandidatePair(
        case_a_id="CS-23224",
        case_b_id="CS-12050",
        reasons=[
            CandidateReason(
                signal="fuzzy_account_match",
                score=96.0,
            ),
            CandidateReason(
                signal="same_contact_email",
                score=1.0,
            ),
        ],
    )

    return cases, case_index, pair


def tool_action(tool, arguments):
    return CallToolAction(
        action="CALL_TOOL",
        tool=tool,
        arguments=arguments,
        reason="Gather the next useful evidence.",
    )


def draft_action(verdict, evidence_ids):
    evidence = []

    for evidence_id in evidence_ids:
        evidence.append(
            {
                "evidence_id": evidence_id,
                "claim": "Claim supported by this tool.",
            }
        )

    recommendation = DraftVerdict(
        verdict=verdict,
        confidence=0.9,
        summary="Recommendation based on tool evidence.",
        evidence=evidence,
        uncertainties=[],
    )

    return DraftVerdictAction(
        action="DRAFT_VERDICT",
        recommendation=recommendation,
    )


def test_model_selects_tools_and_state_accumulates():
    cases, case_index, pair = load_investigation_data()
    llm_client = ScriptedLlmClient(
        [
            tool_action(
                ToolName.FUZZY_SCORE,
                {"field_name": "account_name"},
            ),
            tool_action(
                ToolName.TIMELINE_GAP,
                {},
            ),
            draft_action(
                VerdictValue.DUPLICATE,
                ["tool-1", "tool-2"],
            ),
        ]
    )

    result = run_investigation(
        "INV-1",
        pair,
        cases,
        case_index,
        llm_client,
    )

    assert result.status == "PENDING_REVIEW"
    assert result.recommendation.verdict == (
        VerdictValue.DUPLICATE
    )
    assert result.steps_used == 3
    assert len(llm_client.states[0]["evidence"]) == 0
    assert len(llm_client.states[1]["evidence"]) == 1
    assert len(llm_client.states[2]["evidence"]) == 2

    tool_events = []

    for event in result.trace:
        if event["event_type"] == "TOOL_CALL":
            tool_events.append(event)

    assert tool_events[0]["payload"]["tool"] == (
        "fuzzy_score"
    )
    assert tool_events[1]["payload"]["tool"] == (
        "timeline_gap"
    )


def test_duplicate_tool_call_is_blocked():
    cases, case_index, pair = load_investigation_data()
    repeated_action = tool_action(
        ToolName.FUZZY_SCORE,
        {"field_name": "account_name"},
    )
    llm_client = ScriptedLlmClient(
        [
            repeated_action,
            repeated_action,
            draft_action(
                VerdictValue.UNSURE,
                ["tool-1"],
            ),
        ]
    )

    result = run_investigation(
        "INV-2",
        pair,
        cases,
        case_index,
        llm_client,
    )

    tool_events = []

    for event in result.trace:
        if event["event_type"] == "TOOL_CALL":
            tool_events.append(event)

    assert tool_events[0]["payload"]["status"] == "success"
    assert tool_events[1]["payload"]["status"] == "blocked"


def test_fabricated_evidence_reference_is_rejected():
    cases, case_index, pair = load_investigation_data()
    llm_client = ScriptedLlmClient(
        [
            tool_action(
                ToolName.TIMELINE_GAP,
                {},
            ),
            draft_action(
                VerdictValue.DUPLICATE,
                ["tool-999"],
            ),
            draft_action(
                VerdictValue.DUPLICATE,
                ["tool-1"],
            ),
        ]
    )

    result = run_investigation(
        "INV-3",
        pair,
        cases,
        case_index,
        llm_client,
    )

    event_types = []

    for event in result.trace:
        event_types.append(event["event_type"])

    assert "VERDICT_REJECTED" in event_types
    assert result.recommendation.verdict == (
        VerdictValue.DUPLICATE
    )
    assert len(
        llm_client.states[2]["validation_feedback"]
    ) == 1


def test_step_limit_forces_unsure():
    cases, case_index, pair = load_investigation_data()
    llm_client = ScriptedLlmClient(
        [
            tool_action(
                ToolName.TIMELINE_GAP,
                {},
            ),
            tool_action(
                ToolName.FUZZY_SCORE,
                {"field_name": "subject"},
            ),
        ]
    )

    result = run_investigation(
        "INV-4",
        pair,
        cases,
        case_index,
        llm_client,
        max_steps=2,
    )

    event_types = []

    for event in result.trace:
        event_types.append(event["event_type"])

    assert result.recommendation.verdict == (
        VerdictValue.UNSURE
    )
    assert result.steps_used == 2
    assert "STEP_LIMIT_REACHED" in event_types


def test_related_case_reference_cannot_leave_pair():
    cases, case_index, pair = load_investigation_data()
    llm_client = ScriptedLlmClient(
        [
            tool_action(
                ToolName.FIND_RELATED_CASES,
                {
                    "reference_case_id": "CS-61645",
                    "match_by": "contact_email",
                    "limit": 10,
                },
            ),
            draft_action(
                VerdictValue.UNSURE,
                [],
            ),
        ]
    )

    result = run_investigation(
        "INV-5",
        pair,
        cases,
        case_index,
        llm_client,
    )

    tool_event = None

    for event in result.trace:
        if event["event_type"] == "TOOL_CALL":
            tool_event = event

    assert tool_event is not None
    assert tool_event["payload"]["status"] == "error"
    assert "current pair" in (
        tool_event["payload"]["result"]["error"]
    )


def test_event_writer_receives_append_only_trace():
    cases, case_index, pair = load_investigation_data()
    events = []
    llm_client = ScriptedLlmClient(
        [
            tool_action(
                ToolName.TIMELINE_GAP,
                {},
            ),
            draft_action(
                VerdictValue.DUPLICATE,
                ["tool-1"],
            ),
        ]
    )

    result = run_investigation(
        "INV-6",
        pair,
        cases,
        case_index,
        llm_client,
        event_writer=events.append,
    )

    assert events == result.trace

    sequence_numbers = []

    for event in events:
        sequence_numbers.append(
            event["sequence_number"]
        )

    assert sequence_numbers == list(
        range(1, len(events) + 1)
    )
