from app.data import DEFAULT_DATA_PATH
from app.db import list_candidate_pairs
from scripts.seed import seed_database


def test_seed_loads_cases_and_candidate_pairs(tmp_path):
    db_path = tmp_path / "seed.db"

    result = seed_database(
        csv_path=DEFAULT_DATA_PATH,
        db_path=db_path,
    )

    assert result["loaded_cases"] == 269
    assert result["generated_candidate_pairs"] == 2529
    assert result["inserted_cases"] == 269
    assert result["inserted_candidate_pairs"] == 2529

    stored_pairs = list_candidate_pairs(db_path)

    assert len(stored_pairs) == 2529
