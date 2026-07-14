import pytest

from app.data import index_cases
from app.data import load_cases
from app.tools import compare_identity_and_context
from app.tools import find_related_cases
from app.tools import fuzzy_score
from app.tools import measure_text_prevalence
from app.tools import timeline_gap


def load_case_data():
    cases = load_cases()
    case_index = index_cases(cases)

    return cases, case_index


def test_compare_identity_and_context_reports_exact_matches():
    cases, case_index = load_case_data()

    result = compare_identity_and_context(
        case_index,
        "CS-23224",
        "CS-12050",
    )

    assert result["fields"]["account_name"]["exact_match"] is False
    assert result["fields"]["contact_email"]["exact_match"] is True
    assert result["fields"]["channel"]["exact_match"] is False


def test_fuzzy_score_handles_typographical_account_error():
    cases, case_index = load_case_data()

    result = fuzzy_score(
        case_index,
        "CS-23224",
        "CS-12050",
        "account_name",
    )

    assert result["score"] > 90.0


def test_fuzzy_score_rejects_unsupported_fields():
    cases, case_index = load_case_data()

    with pytest.raises(ValueError):
        fuzzy_score(
            case_index,
            "CS-23224",
            "CS-12050",
            "priority",
        )


def test_timeline_gap_returns_chronological_evidence():
    cases, case_index = load_case_data()

    result = timeline_gap(
        case_index,
        "CS-60493",
        "CS-64238",
    )

    assert result["available"] is True
    assert result["earlier_case_id"] == "CS-60493"
    assert result["gap_hours"] == 9.0


def test_measure_text_prevalence_detects_boilerplate():
    cases, case_index = load_case_data()

    result = measure_text_prevalence(
        cases,
        case_index,
        "CS-61645",
        "CS-28714",
        "subject",
    )

    assert result["same_normalized_text"] is True
    assert result["case_a"]["exact_match_count"] >= 10
    assert result["case_a"]["distinct_account_count"] >= 5
    assert result["case_a"]["likely_boilerplate"] is True


def test_find_related_cases_uses_contact_email():
    cases, case_index = load_case_data()

    result = find_related_cases(
        cases,
        case_index,
        "CS-56388",
        "contact_email",
    )

    related_case_ids = set()

    for related_case in result["cases"]:
        related_case_ids.add(related_case["case_id"])

    assert "CS-13442" in related_case_ids
