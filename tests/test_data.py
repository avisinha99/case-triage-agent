from datetime import datetime

from app.data import index_cases
from app.data import load_cases
from app.data import normalize_email
from app.data import normalize_text


def test_load_cases_reads_all_records():
    cases = load_cases()

    assert len(cases) == 269
    assert cases[0].case_id == "CS-20358"
    assert cases[0].created_at == datetime(
        2026,
        4,
        15,
        17,
        55,
        40,
    )


def test_load_cases_handles_multiline_descriptions():
    cases = load_cases()
    cases_by_id = index_cases(cases)

    case = cases_by_id["CS-51220"]

    assert "Sent from my mobile device" in case.description


def test_load_cases_handles_missing_and_uppercase_emails():
    cases = load_cases()
    cases_by_id = index_cases(cases)

    assert cases_by_id["CS-23022"].contact_email is None
    assert (
        cases_by_id["CS-20828"].contact_email
        == "ingrid.tanaka@ledgewickrealt.example"
    )


def test_load_cases_allows_missing_optional_values(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text(
        (
            "case_id,created_at,channel,status,priority,"
            "account_name,contact_name,contact_email,"
            "subject,description\n"
            "CS-TEST,,,,,,,,,\n"
        ),
        encoding="utf-8",
    )

    cases = load_cases(csv_path)

    assert len(cases) == 1
    assert cases[0].case_id == "CS-TEST"
    assert cases[0].created_at is None
    assert cases[0].account_name == ""
    assert cases[0].contact_email is None


def test_normalize_text_handles_case_spacing_and_punctuation():
    value = normalize_text("  Ostara ENERGY!  ")

    assert value == "ostara energy"


def test_normalize_email_handles_blank_and_mixed_case_values():
    assert normalize_email(" USER@EXAMPLE.COM ") == "user@example.com"
    assert normalize_email("   ") is None
    assert normalize_email(None) is None
