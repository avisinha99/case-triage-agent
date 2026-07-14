from app.candidates import generate_candidate_pairs
from app.candidates import subject_tokens
from app.data import load_cases


def find_pair(candidate_pairs, case_a_id, case_b_id):
    requested_ids = {case_a_id, case_b_id}

    for candidate_pair in candidate_pairs:
        pair_ids = {
            candidate_pair.case_a_id,
            candidate_pair.case_b_id,
        }

        if pair_ids == requested_ids:
            return candidate_pair

    raise AssertionError(
        "Candidate pair was not generated: {0}, {1}".format(
            case_a_id,
            case_b_id,
        )
    )


def reason_signals(candidate_pair):
    signals = set()

    for reason in candidate_pair.reasons:
        signals.add(reason.signal)

    return signals


def test_generates_pair_for_fuzzy_account_and_same_email():
    cases = load_cases()
    candidate_pairs = generate_candidate_pairs(cases)

    candidate_pair = find_pair(
        candidate_pairs,
        "CS-23224",
        "CS-12050",
    )
    signals = reason_signals(candidate_pair)

    assert "fuzzy_account_match" in signals
    assert "same_contact_email" in signals


def test_generates_reworded_follow_up_from_same_contact():
    cases = load_cases()
    candidate_pairs = generate_candidate_pairs(cases)

    candidate_pair = find_pair(
        candidate_pairs,
        "CS-13442",
        "CS-56388",
    )
    signals = reason_signals(candidate_pair)

    assert "same_contact_email" in signals


def test_keeps_cross_account_boilerplate_collision_as_candidate():
    cases = load_cases()
    candidate_pairs = generate_candidate_pairs(cases)

    candidate_pair = find_pair(
        candidate_pairs,
        "CS-61645",
        "CS-28714",
    )
    signals = reason_signals(candidate_pair)

    assert "subject_token_overlap" in signals
    assert "same_contact_email" not in signals


def test_subject_tokens_remove_common_stop_words():
    tokens = subject_tokens(
        "Question about the upcoming maintenance window"
    )

    assert tokens == {
        "question",
        "upcoming",
        "maintenance",
        "window",
    }
