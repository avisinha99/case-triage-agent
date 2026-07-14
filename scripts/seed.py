import argparse
from pathlib import Path

from app.candidates import generate_candidate_pairs
from app.data import DEFAULT_DATA_PATH
from app.data import load_cases
from app.db import DEFAULT_DB_PATH
from app.db import initialize_database
from app.db import insert_candidate_pairs
from app.db import insert_cases


def seed_database(
    csv_path: Path,
    db_path: Path,
) -> dict:
    cases = load_cases(csv_path)
    candidate_pairs = generate_candidate_pairs(cases)

    initialize_database(db_path)

    inserted_cases = insert_cases(cases, db_path)
    inserted_pairs = insert_candidate_pairs(
        candidate_pairs,
        db_path,
    )

    return {
        "loaded_cases": len(cases),
        "generated_candidate_pairs": len(
            candidate_pairs
        ),
        "inserted_cases": inserted_cases,
        "inserted_candidate_pairs": inserted_pairs,
        "database_path": str(db_path),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Load support cases and generate candidate pairs."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
    )

    return parser.parse_args()


def main():
    arguments = parse_arguments()
    result = seed_database(
        arguments.csv,
        arguments.db,
    )

    for key, value in result.items():
        print("{0}: {1}".format(key, value))


if __name__ == "__main__":
    main()
