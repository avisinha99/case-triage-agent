from fastapi.testclient import TestClient

from app.candidates import CandidatePair
from app.candidates import CandidateReason
from app.data import load_cases
from app.db import initialize_database
from app.db import insert_candidate_pairs
from app.db import insert_cases
from app.llm_client import ModelAttempt
from app.llm_client import ModelResult
from app.main import app
from app.main import get_db_path
from app.main import get_llm_client
from app.schema import CallToolAction
from app.schema import DraftVerdict
from app.schema import DraftVerdictAction
from app.schema import ToolName
from app.schema import VerdictValue


class ApiTestLlmClient:
    def __init__(self):
        self.model = "api-test-model"
        self.temperature = 0.1
        self.call_count = 0
        self.actions = [
            CallToolAction(
                action="CALL_TOOL",
                tool=ToolName.TIMELINE_GAP,
                arguments={},
                reason="Check whether the cases are close in time.",
            ),
            DraftVerdictAction(
                action="DRAFT_VERDICT",
                recommendation=DraftVerdict(
                    verdict=VerdictValue.DUPLICATE,
                    confidence=0.9,
                    summary=(
                        "The cases are likely duplicates."
                    ),
                    evidence=[
                        {
                            "evidence_id": "tool-1",
                            "claim": (
                                "The timeline supports a "
                                "follow-up."
                            ),
                        }
                    ],
                    uncertainties=[],
                ),
            ),
        ]

    def request_action(self, state):
        self.call_count += 1
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


def setup_api_database(tmp_path):
    db_path = tmp_path / "api.db"
    initialize_database(db_path)

    cases = load_cases()[:2]
    insert_cases(cases, db_path)

    pair = CandidatePair(
        case_a_id=cases[0].case_id,
        case_b_id=cases[1].case_id,
        reasons=[
            CandidateReason(
                signal="subject_token_overlap",
                score=1.0,
            )
        ],
    )
    insert_candidate_pairs([pair], db_path)

    return db_path


def test_api_runs_reviews_and_finalizes_investigation(
    tmp_path,
):
    db_path = setup_api_database(tmp_path)
    llm_client = ApiTestLlmClient()

    app.dependency_overrides[get_db_path] = (
        lambda: db_path
    )
    app.dependency_overrides[get_llm_client] = (
        lambda: llm_client
    )

    try:
        with TestClient(app) as client:
            pair_response = client.get(
                "/candidate-pairs?limit=10"
            )

            assert pair_response.status_code == 200
            pair_id = pair_response.json()[0]["id"]

            investigation_response = client.post(
                "/candidate-pairs/{0}/investigate".format(
                    pair_id
                )
            )

            assert investigation_response.status_code == 200

            investigation = investigation_response.json()
            investigation_id = investigation["id"]

            assert investigation["status"] == (
                "PENDING_REVIEW"
            )
            assert investigation["final_verdict"] is None

            event_types = []

            for event in investigation["trace"]:
                event_types.append(event["event_type"])

            assert "MODEL_ATTEMPT" in event_types
            assert "TOOL_CALL" in event_types
            assert "DRAFT_VERDICT" in event_types

            pending_response = client.get(
                "/investigations"
            )

            assert pending_response.status_code == 200
            assert len(pending_response.json()) == 1

            repeated_response = client.post(
                "/candidate-pairs/{0}/investigate".format(
                    pair_id
                )
            )

            assert repeated_response.status_code == 200
            assert llm_client.call_count == 2

            invalid_decision = client.post(
                (
                    "/investigations/{0}/decision"
                ).format(investigation_id),
                json={
                    "decision": "APPROVE",
                    "reviewer": "analyst@example.com",
                    "override_verdict": "NOT_DUPLICATE",
                },
            )

            assert invalid_decision.status_code == 422

            decision_response = client.post(
                (
                    "/investigations/{0}/decision"
                ).format(investigation_id),
                json={
                    "decision": "APPROVE",
                    "reviewer": "analyst@example.com",
                    "notes": "Evidence reviewed.",
                },
            )

            assert decision_response.status_code == 200

            finalized = decision_response.json()

            assert finalized["status"] == "FINALIZED"
            assert finalized["final_verdict"] == "DUPLICATE"
            assert finalized["trace"][-1]["event_type"] == (
                "HUMAN_DECISION"
            )

            second_decision = client.post(
                (
                    "/investigations/{0}/decision"
                ).format(investigation_id),
                json={
                    "decision": "REJECT",
                    "reviewer": "other@example.com",
                },
            )

            assert second_decision.status_code == 409
    finally:
        app.dependency_overrides.clear()
