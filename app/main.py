import sqlite3
from pathlib import Path

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Query

from app.agent import DEFAULT_MAX_STEPS
from app.agent import run_investigation
from app.candidates import CandidatePair
from app.candidates import CandidateReason
from app.data import index_cases
from app.data import load_cases
from app.db import DEFAULT_DB_PATH
from app.db import claim_investigation
from app.db import create_event_writer
from app.db import create_investigation
from app.db import get_candidate_pair
from app.db import get_investigation
from app.db import get_investigation_for_pair
from app.db import get_trace
from app.db import initialize_database
from app.db import list_candidate_pairs
from app.db import list_investigations
from app.db import mark_investigation_failed
from app.db import record_human_decision
from app.db import save_draft_recommendation
from app.llm_client import GroqAgentClient
from app.schema import HumanDecisionRequest


app = FastAPI(
    title="Case Triage AI Agent",
    version="1.0.0",
)

CASES = load_cases()
CASE_INDEX = index_cases(CASES)

initialize_database(DEFAULT_DB_PATH)


def get_db_path() -> Path:
    return DEFAULT_DB_PATH


def get_case_data():
    return CASES, CASE_INDEX


def get_llm_client():
    return GroqAgentClient()


def candidate_pair_from_record(
    pair_record: dict,
) -> CandidatePair:
    reasons = []

    for reason in pair_record["reasons"]:
        reasons.append(
            CandidateReason(
                signal=reason["signal"],
                score=reason["score"],
            )
        )

    return CandidatePair(
        case_a_id=pair_record["case_a_id"],
        case_b_id=pair_record["case_b_id"],
        reasons=reasons,
    )


def investigation_with_trace(
    investigation: dict,
    db_path: Path,
) -> dict:
    result = dict(investigation)
    result["trace"] = get_trace(
        investigation["id"],
        db_path,
    )

    return result


@app.get("/health")
def health(
    db_path: Path = Depends(get_db_path),
):
    pairs = list_candidate_pairs(
        db_path,
        limit=1,
    )

    return {
        "status": "ok",
        "database_seeded": len(pairs) > 0,
    }


@app.get("/candidate-pairs")
def candidate_pairs(
    limit: int = Query(default=10, ge=1, le=100),
    db_path: Path = Depends(get_db_path),
):
    return list_candidate_pairs(
        db_path,
        limit=limit,
    )


@app.post("/candidate-pairs/{candidate_pair_id}/investigate")
def investigate_candidate_pair(
    candidate_pair_id: int,
    db_path: Path = Depends(get_db_path),
    case_data=Depends(get_case_data),
    llm_client=Depends(get_llm_client),
):
    pair_record = get_candidate_pair(
        candidate_pair_id,
        db_path,
    )

    if pair_record is None:
        raise HTTPException(
            status_code=404,
            detail="Candidate pair not found",
        )

    investigation = get_investigation_for_pair(
        candidate_pair_id,
        db_path,
    )

    if investigation is None:
        try:
            investigation_id = create_investigation(
                candidate_pair_id,
                DEFAULT_MAX_STEPS,
                db_path,
            )
        except sqlite3.IntegrityError:
            investigation = get_investigation_for_pair(
                candidate_pair_id,
                db_path,
            )

            if investigation is None:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Investigation could not be created"
                    ),
                )

            investigation_id = investigation["id"]
    else:
        investigation_id = investigation["id"]

        if investigation["status"] == "RUNNING":
            raise HTTPException(
                status_code=409,
                detail="Investigation is already running",
            )

        if investigation["status"] != "CREATED":
            return investigation_with_trace(
                investigation,
                db_path,
            )

    claimed = claim_investigation(
        investigation_id,
        db_path,
    )

    if not claimed:
        raise HTTPException(
            status_code=409,
            detail="Investigation could not be claimed",
        )

    cases, case_index = case_data
    pair = candidate_pair_from_record(pair_record)
    event_writer = create_event_writer(db_path)

    try:
        result = run_investigation(
            investigation_id,
            pair,
            cases,
            case_index,
            llm_client,
            max_steps=DEFAULT_MAX_STEPS,
            event_writer=event_writer,
        )
        save_draft_recommendation(
            investigation_id,
            result.recommendation,
            result.steps_used,
            db_path,
        )
    except Exception as error:
        mark_investigation_failed(
            investigation_id,
            db_path,
        )
        raise HTTPException(
            status_code=502,
            detail="Investigation execution failed",
        ) from error

    investigation = get_investigation(
        investigation_id,
        db_path,
    )

    return investigation_with_trace(
        investigation,
        db_path,
    )


@app.get("/investigations")
def investigations(
    status: str = Query(
        default="PENDING_REVIEW"
    ),
    db_path: Path = Depends(get_db_path),
):
    try:
        return list_investigations(
            db_path,
            status=status,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error


@app.get("/investigations/{investigation_id}")
def investigation_detail(
    investigation_id: str,
    db_path: Path = Depends(get_db_path),
):
    investigation = get_investigation(
        investigation_id,
        db_path,
    )

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    return investigation_with_trace(
        investigation,
        db_path,
    )


@app.get("/investigations/{investigation_id}/trace")
def investigation_trace(
    investigation_id: str,
    db_path: Path = Depends(get_db_path),
):
    investigation = get_investigation(
        investigation_id,
        db_path,
    )

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    return get_trace(investigation_id, db_path)


@app.post("/investigations/{investigation_id}/decision")
def decide_investigation(
    investigation_id: str,
    request: HumanDecisionRequest,
    db_path: Path = Depends(get_db_path),
):
    investigation = get_investigation(
        investigation_id,
        db_path,
    )

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation not found",
        )

    override_verdict = None

    if request.override_verdict is not None:
        override_verdict = (
            request.override_verdict.value
        )

    try:
        record_human_decision(
            investigation_id,
            request.decision.value,
            request.reviewer,
            notes=request.notes,
            override_verdict=override_verdict,
            db_path=db_path,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        ) from error

    investigation = get_investigation(
        investigation_id,
        db_path,
    )

    return investigation_with_trace(
        investigation,
        db_path,
    )
