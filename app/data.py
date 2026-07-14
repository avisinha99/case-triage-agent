import csv
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "support_cases.csv"

EXPECTED_COLUMNS = [
    "case_id",
    "created_at",
    "channel",
    "status",
    "priority",
    "account_name",
    "contact_name",
    "contact_email",
    "subject",
    "description",
]


@dataclass(frozen=True)
class Case:
    case_id: str
    created_at: Optional[datetime]
    channel: str
    status: str
    priority: str
    account_name: str
    contact_name: str
    contact_email: Optional[str]
    subject: str
    description: str


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.casefold()

    characters = []

    for character in normalized:
        if character.isalnum():
            characters.append(character)
        else:
            characters.append(" ")

    normalized = "".join(characters)
    return " ".join(normalized.split())


def normalize_email(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    normalized = value.strip().casefold()

    if normalized == "":
        return None

    return normalized


def optional_text(row: dict, field_name: str) -> str:
    value = row.get(field_name)

    if value is None:
        return ""

    return value.strip()


def require_case_id(row: dict, row_number: int) -> str:
    case_id = optional_text(row, "case_id")

    if case_id == "":
        message = "Missing case_id at CSV row {0}"
        raise ValueError(message.format(row_number))

    return case_id


def parse_created_at(
    value: str,
    row_number: int,
) -> Optional[datetime]:
    if value == "":
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        message = "Invalid created_at value at CSV row {0}: {1}"
        raise ValueError(message.format(row_number, value)) from error


def validate_columns(fieldnames: Optional[list[str]]) -> None:
    if fieldnames is None:
        raise ValueError("CSV file does not contain a header row")

    missing_columns = []

    for column in EXPECTED_COLUMNS:
        if column not in fieldnames:
            missing_columns.append(column)

    if missing_columns:
        message = "CSV is missing columns: {0}"
        raise ValueError(message.format(", ".join(missing_columns)))


def load_cases(path: Path = DEFAULT_DATA_PATH) -> list[Case]:
    cases = []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_columns(reader.fieldnames)

        for row_number, row in enumerate(reader, start=2):
            created_at_text = optional_text(row, "created_at")

            case = Case(
                case_id=require_case_id(row, row_number),
                created_at=parse_created_at(
                    created_at_text,
                    row_number,
                ),
                channel=optional_text(row, "channel"),
                status=optional_text(row, "status"),
                priority=optional_text(row, "priority"),
                account_name=optional_text(row, "account_name"),
                contact_name=optional_text(row, "contact_name"),
                contact_email=normalize_email(
                    row.get("contact_email")
                ),
                subject=optional_text(row, "subject"),
                description=optional_text(row, "description"),
            )
            cases.append(case)

    return cases


def index_cases(cases: list[Case]) -> dict[str, Case]:
    case_index = {}

    for case in cases:
        if case.case_id in case_index:
            message = "Duplicate case_id found: {0}"
            raise ValueError(message.format(case.case_id))

        case_index[case.case_id] = case

    return case_index
