import copy
import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone

from app.candidates import CandidatePair
from app.data import Case
from app.llm_client import PROMPT_VERSION
from app.schema import CallToolAction
from app.schema import DraftVerdict
from app.schema import DraftVerdictAction
from app.schema import ToolName
from app.schema import VerdictValue
from app.tools import compare_identity_and_context
from app.tools import find_related_cases
from app.tools import fuzzy_score
from app.tools import get_case
from app.tools import measure_text_prevalence
from app.tools import timeline_gap


DEFAULT_MAX_STEPS = 6


@dataclass
class InvestigationState:
    investigation_id: str
    pair: CandidatePair
    max_steps: int
    step: int = 0
    evidence: list[dict] = field(default_factory=list)
    validation_feedback: list[str] = field(
        default_factory=list
    )
    executed_tool_calls: set[str] = field(
        default_factory=set
    )
    trace: list[dict] = field(default_factory=list)
    event_sequence: int = 0


@dataclass
class InvestigationResult:
    investigation_id: str
    recommendation: DraftVerdict
    trace: list[dict]
    steps_used: int
    status: str = "PENDING_REVIEW"


def case_to_model_data(case: Case) -> dict:
    created_at = None

    if case.created_at is not None:
        created_at = case.created_at.isoformat(sep=" ")

    return {
        "case_id": case.case_id,
        "created_at": created_at,
        "channel": case.channel,
        "status": case.status,
        "priority": case.priority,
        "account_name": case.account_name,
        "contact_name": case.contact_name,
        "contact_email": case.contact_email,
        "subject": case.subject,
        "description": case.description,
    }


def build_model_state(
    state: InvestigationState,
    case_index: dict[str, Case],
) -> dict:
    case_a = get_case(
        case_index,
        state.pair.case_a_id,
    )
    case_b = get_case(
        case_index,
        state.pair.case_b_id,
    )

    candidate_reasons = []

    for reason in state.pair.reasons:
        candidate_reasons.append(
            {
                "signal": reason.signal,
                "score": reason.score,
            }
        )

    return {
        "investigation_id": state.investigation_id,
        "pair": {
            "case_a": case_to_model_data(case_a),
            "case_b": case_to_model_data(case_b),
        },
        "candidate_reasons": candidate_reasons,
        "evidence": copy.deepcopy(state.evidence),
        "validation_feedback": list(
            state.validation_feedback
        ),
        "step": state.step,
        "max_steps": state.max_steps,
        "steps_remaining": (
            state.max_steps - state.step + 1
        ),
    }


def record_event(
    state: InvestigationState,
    event_type: str,
    payload: dict,
    event_writer=None,
) -> dict:
    state.event_sequence += 1

    event = {
        "event_id": "event-{0}".format(
            state.event_sequence
        ),
        "investigation_id": state.investigation_id,
        "sequence_number": state.event_sequence,
        "event_type": event_type,
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "payload": payload,
    }

    state.trace.append(event)

    if event_writer is not None:
        event_writer(event)

    return event


def run_compare_identity(
    cases: list[Case],
    case_index: dict[str, Case],
    pair: CandidatePair,
    arguments: dict,
) -> dict:
    return compare_identity_and_context(
        case_index,
        pair.case_a_id,
        pair.case_b_id,
    )


def run_fuzzy_score(
    cases: list[Case],
    case_index: dict[str, Case],
    pair: CandidatePair,
    arguments: dict,
) -> dict:
    return fuzzy_score(
        case_index,
        pair.case_a_id,
        pair.case_b_id,
        arguments["field_name"],
    )


def run_timeline_gap(
    cases: list[Case],
    case_index: dict[str, Case],
    pair: CandidatePair,
    arguments: dict,
) -> dict:
    return timeline_gap(
        case_index,
        pair.case_a_id,
        pair.case_b_id,
    )


def run_text_prevalence(
    cases: list[Case],
    case_index: dict[str, Case],
    pair: CandidatePair,
    arguments: dict,
) -> dict:
    return measure_text_prevalence(
        cases,
        case_index,
        pair.case_a_id,
        pair.case_b_id,
        arguments["field_name"],
    )


def run_find_related_cases(
    cases: list[Case],
    case_index: dict[str, Case],
    pair: CandidatePair,
    arguments: dict,
) -> dict:
    reference_case_id = arguments["reference_case_id"]
    pair_case_ids = {
        pair.case_a_id,
        pair.case_b_id,
    }

    if reference_case_id not in pair_case_ids:
        raise ValueError(
            "reference_case_id must belong to the current pair"
        )

    return find_related_cases(
        cases,
        case_index,
        reference_case_id,
        arguments["match_by"],
        arguments["limit"],
    )


TOOL_REGISTRY = {
    ToolName.COMPARE_IDENTITY_AND_CONTEXT: (
        run_compare_identity
    ),
    ToolName.FUZZY_SCORE: run_fuzzy_score,
    ToolName.TIMELINE_GAP: run_timeline_gap,
    ToolName.MEASURE_TEXT_PREVALENCE: (
        run_text_prevalence
    ),
    ToolName.FIND_RELATED_CASES: run_find_related_cases,
}


def tool_call_key(action: CallToolAction) -> str:
    return json.dumps(
        {
            "tool": action.tool.value,
            "arguments": action.arguments,
        },
        sort_keys=True,
    )


def execute_tool_action(
    action: CallToolAction,
    state: InvestigationState,
    cases: list[Case],
    case_index: dict[str, Case],
) -> dict:
    call_key = tool_call_key(action)

    if call_key in state.executed_tool_calls:
        return {
            "status": "blocked",
            "result": {
                "error": (
                    "Duplicate tool call with identical "
                    "arguments was blocked."
                )
            },
        }

    state.executed_tool_calls.add(call_key)
    tool_function = TOOL_REGISTRY[action.tool]

    try:
        result = tool_function(
            cases,
            case_index,
            state.pair,
            action.arguments,
        )
    except ValueError as error:
        return {
            "status": "error",
            "result": {
                "error": str(error),
            },
        }

    return {
        "status": "success",
        "result": result,
    }


def record_model_attempts(
    state: InvestigationState,
    model_result,
    llm_client,
    event_writer=None,
) -> None:
    for attempt in model_result.attempts:
        record_event(
            state,
            "MODEL_ATTEMPT",
            {
                "agent_step": state.step,
                "attempt_number": attempt.attempt_number,
                "status": attempt.status,
                "raw_response": attempt.raw_response,
                "error": attempt.error,
                "model": getattr(
                    llm_client,
                    "model",
                    "unknown",
                ),
                "temperature": getattr(
                    llm_client,
                    "temperature",
                    None,
                ),
                "prompt_version": PROMPT_VERSION,
            },
            event_writer,
        )


def successful_evidence_ids(
    state: InvestigationState,
) -> set[str]:
    evidence_ids = set()

    for evidence in state.evidence:
        if evidence["status"] == "success":
            evidence_ids.add(evidence["evidence_id"])

    return evidence_ids


def validate_draft_references(
    recommendation: DraftVerdict,
    state: InvestigationState,
    model_action_succeeded: bool,
) -> tuple[bool, str]:
    valid_evidence_ids = successful_evidence_ids(state)

    if (
        recommendation.verdict == VerdictValue.UNSURE
        and not state.evidence
        and model_action_succeeded
    ):
        return (
            False,
            "Use at least one investigation tool before "
            "drafting an LLM-generated UNSURE verdict.",
        )

    for evidence_reference in recommendation.evidence:
        if (
            evidence_reference.evidence_id
            not in valid_evidence_ids
        ):
            message = (
                "Evidence ID {0} does not reference a "
                "successful tool result."
            )
            return (
                False,
                message.format(
                    evidence_reference.evidence_id
                ),
            )

    decided_verdicts = {
        VerdictValue.DUPLICATE,
        VerdictValue.NOT_DUPLICATE,
    }

    if (
        recommendation.verdict in decided_verdicts
        and not valid_evidence_ids
    ):
        return (
            False,
            "A decided verdict requires a successful "
            "tool result.",
        )

    return True, ""


def step_limit_verdict() -> DraftVerdict:
    return DraftVerdict(
        verdict=VerdictValue.UNSURE,
        confidence=0.0,
        summary=(
            "The investigation reached its maximum "
            "step count without a validated verdict."
        ),
        evidence=[],
        uncertainties=[
            "The bounded investigation ended before "
            "the evidence was sufficient."
        ],
    )


def finish_investigation(
    state: InvestigationState,
    recommendation: DraftVerdict,
    event_writer=None,
) -> InvestigationResult:
    record_event(
        state,
        "DRAFT_VERDICT",
        {
            "status": "PENDING_REVIEW",
            "recommendation": recommendation.model_dump(
                mode="json"
            ),
        },
        event_writer,
    )

    return InvestigationResult(
        investigation_id=state.investigation_id,
        recommendation=recommendation,
        trace=state.trace,
        steps_used=state.step,
    )


def run_investigation(
    investigation_id: str,
    pair: CandidatePair,
    cases: list[Case],
    case_index: dict[str, Case],
    llm_client,
    max_steps: int = DEFAULT_MAX_STEPS,
    event_writer=None,
) -> InvestigationResult:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    get_case(case_index, pair.case_a_id)
    get_case(case_index, pair.case_b_id)

    state = InvestigationState(
        investigation_id=investigation_id,
        pair=pair,
        max_steps=max_steps,
    )

    record_event(
        state,
        "INVESTIGATION_STARTED",
        {
            "case_a_id": pair.case_a_id,
            "case_b_id": pair.case_b_id,
            "max_steps": max_steps,
        },
        event_writer,
    )

    while state.step < state.max_steps:
        state.step += 1
        model_state = build_model_state(
            state,
            case_index,
        )
        model_result = llm_client.request_action(
            model_state
        )

        record_model_attempts(
            state,
            model_result,
            llm_client,
            event_writer,
        )

        action = model_result.action

        if isinstance(action, CallToolAction):
            tool_output = execute_tool_action(
                action,
                state,
                cases,
                case_index,
            )

            evidence_id = "tool-{0}".format(
                len(state.evidence) + 1
            )
            evidence = {
                "evidence_id": evidence_id,
                "tool": action.tool.value,
                "arguments": action.arguments,
                "reason": action.reason,
                "status": tool_output["status"],
                "result": tool_output["result"],
            }
            state.evidence.append(evidence)

            record_event(
                state,
                "TOOL_CALL",
                evidence,
                event_writer,
            )
            continue

        model_action_succeeded = False

        for attempt in model_result.attempts:
            if attempt.status == "success":
                model_action_succeeded = True

        is_valid, validation_error = (
            validate_draft_references(
                action.recommendation,
                state,
                model_action_succeeded,
            )
        )

        if not is_valid:
            state.validation_feedback.append(
                validation_error
            )
            record_event(
                state,
                "VERDICT_REJECTED",
                {
                    "reason": validation_error,
                    "recommendation": (
                        action.recommendation.model_dump(
                            mode="json"
                        )
                    ),
                },
                event_writer,
            )
            continue

        return finish_investigation(
            state,
            action.recommendation,
            event_writer,
        )

    recommendation = step_limit_verdict()

    record_event(
        state,
        "STEP_LIMIT_REACHED",
        {
            "max_steps": state.max_steps,
        },
        event_writer,
    )

    return finish_investigation(
        state,
        recommendation,
        event_writer,
    )
