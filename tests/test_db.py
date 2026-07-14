import sqlite3

import pytest

from app.candidates import CandidatePair
from app.candidates import CandidateReason
from app.data import load_cases
from app.db import append_trace_event
from app.db import claim_investigation
from app.db import connect_database
from app.db import create_investigation
from app.db import get_investigation
from app.db import get_trace
from app.db import initialize_database
from app.db import insert_candidate_pairs
from app.db import insert_cases
from app.db import list_candidate_pairs
from app.db import record_human_decision
from app.db import save_draft_recommendation
from app.schema import DraftVerdict
from app.schema import VerdictValue


def setup_database(tmp_path):
    db_path = tmp_path / "test.db"
    initialize_database(db_path)

    cases = load_cases()[:5]
    insert_cases(cases, db_path)

    pairs = []

    for case in cases[1:]:
        pairs.append(
            CandidatePair(
                case_a_id=cases[0].case_id,
                case_b_id=case.case_id,
                reasons=[
                    CandidateReason(
                        signal="test_signal",
                        score=1.0,
                    )
                ],
            )
        )

    insert_candidate_pairs(pairs, db_path)
    stored_pairs = list_candidate_pairs(db_path)

    return db_path, cases, pairs, stored_pairs


def sample_draft():
    return DraftVerdict(
        verdict=VerdictValue.DUPLICATE,
        confidence=0.9,
        summary="The cases have matching evidence.",
        evidence=[
            {
                "evidence_id": "tool-1",
                "claim": "The identity fields match.",
            }
        ],
        uncertainties=[],
    )


def create_pending_investigation(
    db_path,
    candidate_pair_id,
    investigation_id,
):
    create_investigation(
        candidate_pair_id,
        6,
        db_path,
        investigation_id,
    )
    claimed = claim_investigation(
        investigation_id,
        db_path,
    )
    assert claimed is True

    save_draft_recommendation(
        investigation_id,
        sample_draft(),
        2,
        db_path,
    )


def test_cases_and_candidate_pairs_are_idempotent(tmp_path):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )

    assert len(stored_pairs) == 4
    assert insert_cases(cases, db_path) == 0
    assert insert_candidate_pairs(pairs, db_path) == 0


def test_candidate_pair_allows_only_one_investigation(
    tmp_path,
):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )
    candidate_pair_id = stored_pairs[0]["id"]

    create_investigation(
        candidate_pair_id,
        6,
        db_path,
        "INV-1",
    )

    with pytest.raises(sqlite3.IntegrityError):
        create_investigation(
            candidate_pair_id,
            6,
            db_path,
            "INV-2",
        )


def test_investigation_claim_is_atomic(tmp_path):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )

    create_investigation(
        stored_pairs[0]["id"],
        6,
        db_path,
        "INV-1",
    )

    assert claim_investigation("INV-1", db_path) is True
    assert claim_investigation("INV-1", db_path) is False


def test_trace_events_are_append_only(tmp_path):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )

    create_investigation(
        stored_pairs[0]["id"],
        6,
        db_path,
        "INV-1",
    )
    claim_investigation("INV-1", db_path)

    event = {
        "event_id": "event-1",
        "investigation_id": "INV-1",
        "sequence_number": 1,
        "event_type": "MODEL_ATTEMPT",
        "created_at": "2026-01-01T00:00:00+00:00",
        "payload": {
            "agent_step": 1,
            "status": "success",
        },
    }
    append_trace_event(event, db_path)

    trace = get_trace("INV-1", db_path)

    assert len(trace) == 1
    assert trace[0]["payload"]["status"] == "success"

    connection = connect_database(db_path)

    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                UPDATE trace_events
                SET event_type = 'CHANGED'
                WHERE investigation_id = 'INV-1'
                """
            )
    finally:
        connection.close()


def test_draft_requires_running_investigation(tmp_path):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )

    create_investigation(
        stored_pairs[0]["id"],
        6,
        db_path,
        "INV-1",
    )

    with pytest.raises(ValueError):
        save_draft_recommendation(
            "INV-1",
            sample_draft(),
            1,
            db_path,
        )

    claim_investigation("INV-1", db_path)
    save_draft_recommendation(
        "INV-1",
        sample_draft(),
        2,
        db_path,
    )

    investigation = get_investigation(
        "INV-1",
        db_path,
    )

    assert investigation["status"] == "PENDING_REVIEW"
    assert investigation["final_verdict"] is None


def test_decision_requires_pending_review(tmp_path):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )

    create_investigation(
        stored_pairs[0]["id"],
        6,
        db_path,
        "INV-1",
    )

    with pytest.raises(ValueError):
        record_human_decision(
            "INV-1",
            "APPROVE",
            "reviewer@example.com",
            db_path=db_path,
        )


def test_approve_finalizes_draft_and_logs_decision(
    tmp_path,
):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )
    create_pending_investigation(
        db_path,
        stored_pairs[0]["id"],
        "INV-1",
    )

    record_human_decision(
        "INV-1",
        "APPROVE",
        "reviewer@example.com",
        notes="Evidence reviewed.",
        db_path=db_path,
    )

    investigation = get_investigation(
        "INV-1",
        db_path,
    )
    trace = get_trace("INV-1", db_path)

    assert investigation["status"] == "FINALIZED"
    assert investigation["final_verdict"] == "DUPLICATE"
    assert trace[-1]["event_type"] == "HUMAN_DECISION"
    assert trace[-1]["payload"]["reviewer"] == (
        "reviewer@example.com"
    )

    with pytest.raises(ValueError):
        record_human_decision(
            "INV-1",
            "REJECT",
            "second-reviewer@example.com",
            db_path=db_path,
        )


def test_override_requires_and_sets_new_verdict(tmp_path):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )
    create_pending_investigation(
        db_path,
        stored_pairs[1]["id"],
        "INV-2",
    )

    with pytest.raises(ValueError):
        record_human_decision(
            "INV-2",
            "OVERRIDE",
            "reviewer@example.com",
            db_path=db_path,
        )

    record_human_decision(
        "INV-2",
        "OVERRIDE",
        "reviewer@example.com",
        override_verdict="NOT_DUPLICATE",
        db_path=db_path,
    )

    investigation = get_investigation(
        "INV-2",
        db_path,
    )

    assert investigation["status"] == "FINALIZED"
    assert investigation["final_verdict"] == (
        "NOT_DUPLICATE"
    )


def test_reject_finalizes_without_accepted_verdict(tmp_path):
    db_path, cases, pairs, stored_pairs = setup_database(
        tmp_path
    )
    create_pending_investigation(
        db_path,
        stored_pairs[2]["id"],
        "INV-3",
    )

    record_human_decision(
        "INV-3",
        "REJECT",
        "reviewer@example.com",
        db_path=db_path,
    )

    investigation = get_investigation(
        "INV-3",
        db_path,
    )

    assert investigation["status"] == "FINALIZED"
    assert investigation["final_verdict"] is None
