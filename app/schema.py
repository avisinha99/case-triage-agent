from enum import Enum
from typing import Any
from typing import Literal
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerdictValue(str, Enum):
    DUPLICATE = "DUPLICATE"
    NOT_DUPLICATE = "NOT_DUPLICATE"
    UNSURE = "UNSURE"


class ToolName(str, Enum):
    COMPARE_IDENTITY_AND_CONTEXT = (
        "compare_identity_and_context"
    )
    FUZZY_SCORE = "fuzzy_score"
    TIMELINE_GAP = "timeline_gap"
    MEASURE_TEXT_PREVALENCE = "measure_text_prevalence"
    FIND_RELATED_CASES = "find_related_cases"


class EvidenceReference(StrictModel):
    evidence_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)


class DraftVerdict(StrictModel):
    verdict: VerdictValue
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1)
    evidence: list[EvidenceReference] = Field(
        default_factory=list
    )
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence_for_decision(self):
        decided_verdicts = {
            VerdictValue.DUPLICATE,
            VerdictValue.NOT_DUPLICATE,
        }

        if (
            self.verdict in decided_verdicts
            and not self.evidence
        ):
            raise ValueError(
                "A decided verdict must cite tool evidence"
            )

        return self


class NoArguments(StrictModel):
    pass


class FuzzyScoreArguments(StrictModel):
    field_name: Literal[
        "account_name",
        "contact_name",
        "subject",
        "description",
    ]


class TextPrevalenceArguments(StrictModel):
    field_name: Literal[
        "subject",
        "description",
    ]


class FindRelatedCasesArguments(StrictModel):
    reference_case_id: str = Field(min_length=1)
    match_by: Literal[
        "account_name",
        "contact_email",
        "contact_name",
    ]
    limit: int = Field(default=10, ge=1, le=50)


class CallToolAction(StrictModel):
    action: Literal["CALL_TOOL"]
    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(min_length=1)


class DraftVerdictAction(StrictModel):
    action: Literal["DRAFT_VERDICT"]
    recommendation: DraftVerdict


AgentAction = Union[
    CallToolAction,
    DraftVerdictAction,
]


TOOL_ARGUMENT_MODELS = {
    ToolName.COMPARE_IDENTITY_AND_CONTEXT: NoArguments,
    ToolName.FUZZY_SCORE: FuzzyScoreArguments,
    ToolName.TIMELINE_GAP: NoArguments,
    ToolName.MEASURE_TEXT_PREVALENCE: (
        TextPrevalenceArguments
    ),
    ToolName.FIND_RELATED_CASES: FindRelatedCasesArguments,
}


def parse_agent_action(payload: dict) -> AgentAction:
    action = payload.get("action")

    if action == "CALL_TOOL":
        return CallToolAction.model_validate(payload)

    if action == "DRAFT_VERDICT":
        return DraftVerdictAction.model_validate(payload)

    raise ValueError("Unknown agent action: {0}".format(action))


def validate_tool_arguments(
    action: CallToolAction,
) -> dict:
    argument_model = TOOL_ARGUMENT_MODELS[action.tool]
    validated_arguments = argument_model.model_validate(
        action.arguments
    )

    return validated_arguments.model_dump()
