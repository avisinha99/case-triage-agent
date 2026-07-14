from dataclasses import dataclass

from rapidfuzz import fuzz

from app.data import Case
from app.data import normalize_text


ACCOUNT_MATCH_THRESHOLD = 85.0
SUBJECT_OVERLAP_THRESHOLD = 0.5

SUBJECT_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


@dataclass
class CandidateReason:
    signal: str
    score: float


@dataclass
class CandidatePair:
    case_a_id: str
    case_b_id: str
    reasons: list[CandidateReason]


def text_similarity(value_a: str, value_b: str) -> float:
    normalized_a = normalize_text(value_a)
    normalized_b = normalize_text(value_b)

    if normalized_a == "" or normalized_b == "":
        return 0.0

    return float(fuzz.ratio(normalized_a, normalized_b))


def account_similarity(case_a: Case, case_b: Case) -> float:
    return text_similarity(
        case_a.account_name,
        case_b.account_name,
    )


def same_contact_email(case_a: Case, case_b: Case) -> bool:
    if case_a.contact_email is None:
        return False

    if case_b.contact_email is None:
        return False

    return case_a.contact_email == case_b.contact_email


def subject_tokens(subject: str) -> set[str]:
    normalized = normalize_text(subject)
    tokens = set()

    for token in normalized.split():
        if token not in SUBJECT_STOP_WORDS:
            tokens.add(token)

    return tokens


def subject_overlap(case_a: Case, case_b: Case) -> float:
    tokens_a = subject_tokens(case_a.subject)
    tokens_b = subject_tokens(case_b.subject)

    if not tokens_a or not tokens_b:
        return 0.0

    shared_tokens = tokens_a.intersection(tokens_b)
    all_tokens = tokens_a.union(tokens_b)

    return len(shared_tokens) / len(all_tokens)


def candidate_reasons(
    case_a: Case,
    case_b: Case,
) -> list[CandidateReason]:
    reasons = []

    account_score = account_similarity(case_a, case_b)

    if account_score >= ACCOUNT_MATCH_THRESHOLD:
        reasons.append(
            CandidateReason(
                signal="fuzzy_account_match",
                score=account_score,
            )
        )

    if same_contact_email(case_a, case_b):
        reasons.append(
            CandidateReason(
                signal="same_contact_email",
                score=1.0,
            )
        )

    subject_score = subject_overlap(case_a, case_b)

    if subject_score >= SUBJECT_OVERLAP_THRESHOLD:
        reasons.append(
            CandidateReason(
                signal="subject_token_overlap",
                score=subject_score,
            )
        )

    return reasons


def generate_candidate_pairs(
    cases: list[Case],
) -> list[CandidatePair]:
    candidate_pairs = []

    for case_a_position in range(len(cases)):
        case_a = cases[case_a_position]

        for case_b_position in range(
            case_a_position + 1,
            len(cases),
        ):
            case_b = cases[case_b_position]
            reasons = candidate_reasons(case_a, case_b)

            if reasons:
                candidate_pairs.append(
                    CandidatePair(
                        case_a_id=case_a.case_id,
                        case_b_id=case_b.case_id,
                        reasons=reasons,
                    )
                )

    return candidate_pairs
