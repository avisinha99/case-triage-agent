from typing import Optional

from app.candidates import ACCOUNT_MATCH_THRESHOLD
from app.candidates import account_similarity
from app.candidates import text_similarity
from app.data import Case
from app.data import normalize_text


FUZZY_FIELDS = {
    "account_name",
    "contact_name",
    "subject",
    "description",
}

PREVALENCE_FIELDS = {
    "subject",
    "description",
}

RELATED_CASE_FIELDS = {
    "account_name",
    "contact_email",
    "contact_name",
}


def get_case(
    case_index: dict[str, Case],
    case_id: str,
) -> Case:
    case = case_index.get(case_id)

    if case is None:
        message = "Unknown case_id: {0}"
        raise ValueError(message.format(case_id))

    return case


def get_pair(
    case_index: dict[str, Case],
    case_a_id: str,
    case_b_id: str,
) -> tuple[Case, Case]:
    if case_a_id == case_b_id:
        raise ValueError("A case cannot be compared with itself")

    case_a = get_case(case_index, case_a_id)
    case_b = get_case(case_index, case_b_id)

    return case_a, case_b


def compare_identity_and_context(
    case_index: dict[str, Case],
    case_a_id: str,
    case_b_id: str,
) -> dict:
    case_a, case_b = get_pair(
        case_index,
        case_a_id,
        case_b_id,
    )

    field_names = [
        "account_name",
        "contact_name",
        "contact_email",
        "channel",
        "status",
        "priority",
    ]

    comparisons = {}

    for field_name in field_names:
        value_a = getattr(case_a, field_name)
        value_b = getattr(case_b, field_name)

        if field_name == "contact_email":
            normalized_a = value_a or ""
            normalized_b = value_b or ""
        else:
            normalized_a = normalize_text(value_a)
            normalized_b = normalize_text(value_b)

        available = (
            normalized_a != ""
            and normalized_b != ""
        )

        exact_match = None

        if available:
            exact_match = normalized_a == normalized_b

        comparisons[field_name] = {
            "case_a_value": value_a,
            "case_b_value": value_b,
            "available": available,
            "exact_match": exact_match,
        }

    return {
        "case_a_id": case_a.case_id,
        "case_b_id": case_b.case_id,
        "fields": comparisons,
    }


def fuzzy_score(
    case_index: dict[str, Case],
    case_a_id: str,
    case_b_id: str,
    field_name: str,
) -> dict:
    if field_name not in FUZZY_FIELDS:
        message = "Unsupported fuzzy field: {0}"
        raise ValueError(message.format(field_name))

    case_a, case_b = get_pair(
        case_index,
        case_a_id,
        case_b_id,
    )

    value_a = getattr(case_a, field_name)
    value_b = getattr(case_b, field_name)

    score = text_similarity(
        value_a,
        value_b,
    )

    return {
        "case_a_id": case_a.case_id,
        "case_b_id": case_b.case_id,
        "field": field_name,
        "score": round(score, 2),
    }


def timeline_gap(
    case_index: dict[str, Case],
    case_a_id: str,
    case_b_id: str,
) -> dict:
    case_a, case_b = get_pair(
        case_index,
        case_a_id,
        case_b_id,
    )

    missing_timestamps = []

    if case_a.created_at is None:
        missing_timestamps.append(case_a.case_id)

    if case_b.created_at is None:
        missing_timestamps.append(case_b.case_id)

    if missing_timestamps:
        return {
            "case_a_id": case_a.case_id,
            "case_b_id": case_b.case_id,
            "available": False,
            "missing_timestamps": missing_timestamps,
        }

    difference = case_b.created_at - case_a.created_at
    total_seconds = abs(difference.total_seconds())

    if case_a.created_at <= case_b.created_at:
        earlier_case_id = case_a.case_id
        later_case_id = case_b.case_id
    else:
        earlier_case_id = case_b.case_id
        later_case_id = case_a.case_id

    return {
        "case_a_id": case_a.case_id,
        "case_b_id": case_b.case_id,
        "available": True,
        "earlier_case_id": earlier_case_id,
        "later_case_id": later_case_id,
        "gap_minutes": round(total_seconds / 60, 2),
        "gap_hours": round(total_seconds / 3600, 2),
        "gap_days": round(total_seconds / 86400, 2),
    }


def calculate_text_prevalence(
    cases: list[Case],
    reference_case: Case,
    field_name: str,
) -> dict:
    reference_value = getattr(reference_case, field_name)
    normalized_reference = normalize_text(reference_value)

    if normalized_reference == "":
        return {
            "case_id": reference_case.case_id,
            "exact_match_count": 0,
            "distinct_account_count": 0,
            "matched_case_ids": [],
            "likely_boilerplate": False,
        }

    matched_case_ids = []
    matched_accounts = set()

    for case in cases:
        candidate_value = getattr(case, field_name)
        normalized_candidate = normalize_text(candidate_value)

        if normalized_candidate == normalized_reference:
            matched_case_ids.append(case.case_id)

            normalized_account = normalize_text(
                case.account_name
            )

            if normalized_account != "":
                matched_accounts.add(normalized_account)

    exact_match_count = len(matched_case_ids)
    distinct_account_count = len(matched_accounts)

    likely_boilerplate = (
        exact_match_count >= 3
        and distinct_account_count >= 3
    )

    return {
        "case_id": reference_case.case_id,
        "exact_match_count": exact_match_count,
        "distinct_account_count": distinct_account_count,
        "matched_case_ids": matched_case_ids,
        "likely_boilerplate": likely_boilerplate,
    }


def measure_text_prevalence(
    cases: list[Case],
    case_index: dict[str, Case],
    case_a_id: str,
    case_b_id: str,
    field_name: str,
) -> dict:
    if field_name not in PREVALENCE_FIELDS:
        message = "Unsupported prevalence field: {0}"
        raise ValueError(message.format(field_name))

    case_a, case_b = get_pair(
        case_index,
        case_a_id,
        case_b_id,
    )

    value_a = getattr(case_a, field_name)
    value_b = getattr(case_b, field_name)

    normalized_a = normalize_text(value_a)
    normalized_b = normalize_text(value_b)

    same_normalized_text = (
        normalized_a != ""
        and normalized_a == normalized_b
    )

    return {
        "field": field_name,
        "same_normalized_text": same_normalized_text,
        "case_a": calculate_text_prevalence(
            cases,
            case_a,
            field_name,
        ),
        "case_b": calculate_text_prevalence(
            cases,
            case_b,
            field_name,
        ),
    }


def calculate_related_match(
    reference_case: Case,
    candidate_case: Case,
    match_by: str,
) -> Optional[float]:
    if match_by == "account_name":
        score = account_similarity(
            reference_case,
            candidate_case,
        )

        if score >= ACCOUNT_MATCH_THRESHOLD:
            return round(score, 2)

        return None

    if match_by == "contact_email":
        if reference_case.contact_email is None:
            return None

        if candidate_case.contact_email is None:
            return None

        if (
            reference_case.contact_email
            == candidate_case.contact_email
        ):
            return 1.0

        return None

    reference_name = normalize_text(
        reference_case.contact_name
    )
    candidate_name = normalize_text(
        candidate_case.contact_name
    )

    if reference_name == "" or candidate_name == "":
        return None

    if reference_name == candidate_name:
        return 1.0

    return None


def time_gap_hours(
    case_a: Case,
    case_b: Case,
) -> Optional[float]:
    if case_a.created_at is None:
        return None

    if case_b.created_at is None:
        return None

    difference = case_b.created_at - case_a.created_at
    total_seconds = abs(difference.total_seconds())

    return round(total_seconds / 3600, 2)


def related_case_sort_value(related_case: dict) -> float:
    gap_hours = related_case["time_gap_hours"]

    if gap_hours is None:
        return float("inf")

    return gap_hours


def find_related_cases(
    cases: list[Case],
    case_index: dict[str, Case],
    reference_case_id: str,
    match_by: str,
    limit: int = 10,
) -> dict:
    if match_by not in RELATED_CASE_FIELDS:
        message = "Unsupported related-case field: {0}"
        raise ValueError(message.format(match_by))

    if limit < 1 or limit > 50:
        raise ValueError("limit must be between 1 and 50")

    reference_case = get_case(
        case_index,
        reference_case_id,
    )

    related_cases = []

    for candidate_case in cases:
        if candidate_case.case_id == reference_case.case_id:
            continue

        match_score = calculate_related_match(
            reference_case,
            candidate_case,
            match_by,
        )

        if match_score is None:
            continue

        created_at = None

        if candidate_case.created_at is not None:
            created_at = candidate_case.created_at.isoformat(
                sep=" "
            )

        related_cases.append(
            {
                "case_id": candidate_case.case_id,
                "created_at": created_at,
                "account_name": candidate_case.account_name,
                "contact_name": candidate_case.contact_name,
                "contact_email": candidate_case.contact_email,
                "channel": candidate_case.channel,
                "status": candidate_case.status,
                "subject": candidate_case.subject,
                "time_gap_hours": time_gap_hours(
                    reference_case,
                    candidate_case,
                ),
                "match_score": match_score,
            }
        )

    related_cases.sort(key=related_case_sort_value)

    return {
        "reference_case_id": reference_case.case_id,
        "match_by": match_by,
        "total_matches": len(related_cases),
        "cases": related_cases[:limit],
    }
