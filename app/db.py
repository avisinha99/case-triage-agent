import json
import sqlite3
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Optional

from app.candidates import CandidatePair
from app.data import Case
from app.data import PROJECT_ROOT
from app.schema import DraftVerdict
from app.schema import VerdictValue


DEFAULT_DB_PATH = PROJECT_ROOT / "case_triage.db"

INVESTIGATION_STATUSES = {
    "CREATED",
    "RUNNING",
    "PENDING_REVIEW",
    "FINALIZED",
    "FAILED",
}

HUMAN_DECISIONS = {
    "APPROVE",
    "REJECT",
    "OVERRIDE",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def connect_database(
    db_path: Path = DEFAULT_DB_PATH,
) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    return connection


def initialize_database(
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    connection = connect_database(db_path)

    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                created_at TEXT,
                channel TEXT NOT NULL,
                status TEXT NOT NULL,
                priority TEXT NOT NULL,
                account_name TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                contact_email TEXT,
                subject TEXT NOT NULL,
                description TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_a_id TEXT NOT NULL,
                case_b_id TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK (case_a_id <> case_b_id),
                UNIQUE (case_a_id, case_b_id),
                FOREIGN KEY (case_a_id) REFERENCES cases(case_id),
                FOREIGN KEY (case_b_id) REFERENCES cases(case_id)
            );

            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                candidate_pair_id INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL,
                current_step INTEGER NOT NULL DEFAULT 0,
                max_steps INTEGER NOT NULL,
                draft_verdict_json TEXT,
                final_verdict TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finalized_at TEXT,
                CHECK (
                    status IN (
                        'CREATED',
                        'RUNNING',
                        'PENDING_REVIEW',
                        'FINALIZED',
                        'FAILED'
                    )
                ),
                CHECK (
                    final_verdict IS NULL
                    OR final_verdict IN (
                        'DUPLICATE',
                        'NOT_DUPLICATE',
                        'UNSURE'
                    )
                ),
                FOREIGN KEY (candidate_pair_id)
                    REFERENCES candidate_pairs(id)
            );

            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                investigation_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE (investigation_id, event_id),
                UNIQUE (investigation_id, sequence_number),
                FOREIGN KEY (investigation_id)
                    REFERENCES investigations(id)
            );

            CREATE TABLE IF NOT EXISTS human_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                investigation_id TEXT NOT NULL UNIQUE,
                decision TEXT NOT NULL,
                override_verdict TEXT,
                reviewer TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                CHECK (
                    decision IN (
                        'APPROVE',
                        'REJECT',
                        'OVERRIDE'
                    )
                ),
                CHECK (
                    override_verdict IS NULL
                    OR override_verdict IN (
                        'DUPLICATE',
                        'NOT_DUPLICATE',
                        'UNSURE'
                    )
                ),
                FOREIGN KEY (investigation_id)
                    REFERENCES investigations(id)
            );

            CREATE TRIGGER IF NOT EXISTS
                prevent_trace_event_update
            BEFORE UPDATE ON trace_events
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'trace events are append-only'
                );
            END;

            CREATE TRIGGER IF NOT EXISTS
                prevent_trace_event_delete
            BEFORE DELETE ON trace_events
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'trace events are append-only'
                );
            END;

            CREATE TRIGGER IF NOT EXISTS
                prevent_human_decision_update
            BEFORE UPDATE ON human_decisions
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'human decisions are append-only'
                );
            END;

            CREATE TRIGGER IF NOT EXISTS
                prevent_human_decision_delete
            BEFORE DELETE ON human_decisions
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'human decisions are append-only'
                );
            END;
            """
        )
        connection.commit()
    finally:
        connection.close()


def insert_cases(
    cases: list[Case],
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    connection = connect_database(db_path)
    inserted_count = 0

    try:
        with connection:
            for case in cases:
                created_at = None

                if case.created_at is not None:
                    created_at = case.created_at.isoformat(
                        sep=" "
                    )

                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO cases (
                        case_id,
                        created_at,
                        channel,
                        status,
                        priority,
                        account_name,
                        contact_name,
                        contact_email,
                        subject,
                        description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case.case_id,
                        created_at,
                        case.channel,
                        case.status,
                        case.priority,
                        case.account_name,
                        case.contact_name,
                        case.contact_email,
                        case.subject,
                        case.description,
                    ),
                )
                inserted_count += cursor.rowcount
    finally:
        connection.close()

    return inserted_count


def insert_candidate_pairs(
    pairs: list[CandidatePair],
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    connection = connect_database(db_path)
    inserted_count = 0

    try:
        with connection:
            for pair in pairs:
                case_ids = sorted(
                    [
                        pair.case_a_id,
                        pair.case_b_id,
                    ]
                )
                reasons = []

                for reason in pair.reasons:
                    reasons.append(
                        {
                            "signal": reason.signal,
                            "score": reason.score,
                        }
                    )

                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO candidate_pairs (
                        case_a_id,
                        case_b_id,
                        reasons_json,
                        created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        case_ids[0],
                        case_ids[1],
                        serialize_json(reasons),
                        utc_now(),
                    ),
                )
                inserted_count += cursor.rowcount
    finally:
        connection.close()

    return inserted_count


def list_candidate_pairs(
    db_path: Path = DEFAULT_DB_PATH,
    limit: Optional[int] = None,
) -> list[dict]:
    connection = connect_database(db_path)

    try:
        query = """
            SELECT id, case_a_id, case_b_id, reasons_json
            FROM candidate_pairs
            ORDER BY id
        """
        arguments = ()

        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be at least 1")

            query += " LIMIT ?"
            arguments = (limit,)

        rows = connection.execute(
            query,
            arguments,
        ).fetchall()

        pairs = []

        for row in rows:
            pairs.append(
                {
                    "id": row["id"],
                    "case_a_id": row["case_a_id"],
                    "case_b_id": row["case_b_id"],
                    "reasons": json.loads(
                        row["reasons_json"]
                    ),
                }
            )

        return pairs
    finally:
        connection.close()


def create_investigation(
    candidate_pair_id: int,
    max_steps: int,
    db_path: Path = DEFAULT_DB_PATH,
    investigation_id: Optional[str] = None,
) -> str:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    if investigation_id is None:
        investigation_id = "INV-{0}".format(
            uuid.uuid4().hex
        )

    now = utc_now()
    connection = connect_database(db_path)

    try:
        with connection:
            connection.execute(
                """
                INSERT INTO investigations (
                    id,
                    candidate_pair_id,
                    status,
                    current_step,
                    max_steps,
                    created_at,
                    updated_at
                ) VALUES (?, ?, 'CREATED', 0, ?, ?, ?)
                """,
                (
                    investigation_id,
                    candidate_pair_id,
                    max_steps,
                    now,
                    now,
                ),
            )
    finally:
        connection.close()

    return investigation_id


def claim_investigation(
    investigation_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> bool:
    connection = connect_database(db_path)

    try:
        with connection:
            cursor = connection.execute(
                """
                UPDATE investigations
                SET status = 'RUNNING',
                    updated_at = ?
                WHERE id = ?
                  AND status = 'CREATED'
                """,
                (
                    utc_now(),
                    investigation_id,
                ),
            )

        return cursor.rowcount == 1
    finally:
        connection.close()


def append_trace_event(
    event: dict,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    connection = connect_database(db_path)

    try:
        with connection:
            connection.execute(
                """
                INSERT INTO trace_events (
                    event_id,
                    investigation_id,
                    sequence_number,
                    event_type,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    event["investigation_id"],
                    event["sequence_number"],
                    event["event_type"],
                    serialize_json(event["payload"]),
                    event["created_at"],
                ),
            )

            agent_step = event["payload"].get(
                "agent_step"
            )

            if isinstance(agent_step, int):
                connection.execute(
                    """
                    UPDATE investigations
                    SET current_step = MAX(
                            current_step,
                            ?
                        ),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        agent_step,
                        utc_now(),
                        event["investigation_id"],
                    ),
                )
    finally:
        connection.close()


def create_event_writer(
    db_path: Path = DEFAULT_DB_PATH,
):
    def write_event(event):
        append_trace_event(event, db_path)

    return write_event


def save_draft_recommendation(
    investigation_id: str,
    recommendation: DraftVerdict,
    steps_used: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    recommendation_json = serialize_json(
        recommendation.model_dump(mode="json")
    )
    connection = connect_database(db_path)

    try:
        with connection:
            cursor = connection.execute(
                """
                UPDATE investigations
                SET status = 'PENDING_REVIEW',
                    current_step = ?,
                    draft_verdict_json = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status = 'RUNNING'
                """,
                (
                    steps_used,
                    recommendation_json,
                    utc_now(),
                    investigation_id,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Draft can only be saved from RUNNING"
                )
    finally:
        connection.close()


def validate_verdict_value(
    verdict: Optional[str],
) -> Optional[str]:
    if verdict is None:
        return None

    allowed_values = {
        VerdictValue.DUPLICATE.value,
        VerdictValue.NOT_DUPLICATE.value,
        VerdictValue.UNSURE.value,
    }

    if verdict not in allowed_values:
        raise ValueError(
            "Invalid verdict value: {0}".format(verdict)
        )

    return verdict


def record_human_decision(
    investigation_id: str,
    decision: str,
    reviewer: str,
    notes: Optional[str] = None,
    override_verdict: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    decision = decision.strip().upper()
    reviewer = reviewer.strip()

    if decision not in HUMAN_DECISIONS:
        raise ValueError(
            "Invalid human decision: {0}".format(decision)
        )

    if reviewer == "":
        raise ValueError("reviewer cannot be empty")

    override_verdict = validate_verdict_value(
        override_verdict
    )

    if decision == "OVERRIDE":
        if override_verdict is None:
            raise ValueError(
                "OVERRIDE requires override_verdict"
            )
    elif override_verdict is not None:
        raise ValueError(
            "override_verdict is only valid for OVERRIDE"
        )

    connection = connect_database(db_path)

    try:
        with connection:
            investigation = connection.execute(
                """
                SELECT status, draft_verdict_json
                FROM investigations
                WHERE id = ?
                """,
                (investigation_id,),
            ).fetchone()

            if investigation is None:
                raise ValueError(
                    "Unknown investigation: {0}".format(
                        investigation_id
                    )
                )

            if investigation["status"] != "PENDING_REVIEW":
                raise ValueError(
                    "Decision requires PENDING_REVIEW status"
                )

            draft = json.loads(
                investigation["draft_verdict_json"]
            )
            final_verdict = None

            if decision == "APPROVE":
                final_verdict = draft["verdict"]
            elif decision == "OVERRIDE":
                final_verdict = override_verdict

            now = utc_now()

            connection.execute(
                """
                INSERT INTO human_decisions (
                    investigation_id,
                    decision,
                    override_verdict,
                    reviewer,
                    notes,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    investigation_id,
                    decision,
                    override_verdict,
                    reviewer,
                    notes,
                    now,
                ),
            )

            sequence_row = connection.execute(
                """
                SELECT COALESCE(
                    MAX(sequence_number),
                    0
                ) + 1 AS next_sequence
                FROM trace_events
                WHERE investigation_id = ?
                """,
                (investigation_id,),
            ).fetchone()

            decision_payload = {
                "decision": decision,
                "override_verdict": override_verdict,
                "reviewer": reviewer,
                "notes": notes,
                "final_verdict": final_verdict,
            }

            connection.execute(
                """
                INSERT INTO trace_events (
                    event_id,
                    investigation_id,
                    sequence_number,
                    event_type,
                    payload_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "human-decision",
                    investigation_id,
                    sequence_row["next_sequence"],
                    "HUMAN_DECISION",
                    serialize_json(decision_payload),
                    now,
                ),
            )

            cursor = connection.execute(
                """
                UPDATE investigations
                SET status = 'FINALIZED',
                    final_verdict = ?,
                    finalized_at = ?,
                    updated_at = ?
                WHERE id = ?
                  AND status = 'PENDING_REVIEW'
                """,
                (
                    final_verdict,
                    now,
                    now,
                    investigation_id,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "Investigation could not be finalized"
                )
    finally:
        connection.close()


def get_investigation(
    investigation_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[dict]:
    connection = connect_database(db_path)

    try:
        row = connection.execute(
            """
            SELECT
                investigations.*,
                candidate_pairs.case_a_id,
                candidate_pairs.case_b_id,
                candidate_pairs.reasons_json
            FROM investigations
            JOIN candidate_pairs
              ON candidate_pairs.id =
                 investigations.candidate_pair_id
            WHERE investigations.id = ?
            """,
            (investigation_id,),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["reasons"] = json.loads(
            result.pop("reasons_json")
        )

        draft_json = result.pop("draft_verdict_json")
        result["draft_verdict"] = None

        if draft_json is not None:
            result["draft_verdict"] = json.loads(
                draft_json
            )

        return result
    finally:
        connection.close()


def list_investigations(
    db_path: Path = DEFAULT_DB_PATH,
    status: Optional[str] = None,
) -> list[dict]:
    connection = connect_database(db_path)

    try:
        query = """
            SELECT id
            FROM investigations
        """
        arguments = ()

        if status is not None:
            if status not in INVESTIGATION_STATUSES:
                raise ValueError(
                    "Invalid investigation status"
                )

            query += " WHERE status = ?"
            arguments = (status,)

        query += " ORDER BY created_at"

        rows = connection.execute(
            query,
            arguments,
        ).fetchall()

        investigations = []

        for row in rows:
            investigation = get_investigation(
                row["id"],
                db_path,
            )

            if investigation is not None:
                investigations.append(investigation)

        return investigations
    finally:
        connection.close()


def get_trace(
    investigation_id: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> list[dict]:
    connection = connect_database(db_path)

    try:
        rows = connection.execute(
            """
            SELECT
                event_id,
                investigation_id,
                sequence_number,
                event_type,
                payload_json,
                created_at
            FROM trace_events
            WHERE investigation_id = ?
            ORDER BY sequence_number
            """,
            (investigation_id,),
        ).fetchall()

        events = []

        for row in rows:
            event = dict(row)
            event["payload"] = json.loads(
                event.pop("payload_json")
            )
            events.append(event)

        return events
    finally:
        connection.close()
