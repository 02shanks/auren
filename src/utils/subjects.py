"""Subject-name canonicalization.

User shorthand must match dataset subjects: a query about a "Maths test" has to match a
test whose subject is stored as "Mathematics". We map common short forms/synonyms to a
canonical key and compare on canonical *words*, so this is data-driven and generalizes to
any subject rather than being hard-coded to one test or student.
"""

# short form / synonym -> canonical subject
_ALIASES = {
    "math": "mathematics",
    "maths": "mathematics",
    "mathematics": "mathematics",
    "sci": "science",
    "science": "science",
    "eng": "english",
    "english": "english",
    "phys": "physics",
    "physics": "physics",
    "chem": "chemistry",
    "chemistry": "chemistry",
    "bio": "biology",
    "biology": "biology",
    "cs": "computer science",
    "comp sci": "computer science",
    "computer science": "computer science",
    "sst": "social science",
    "social studies": "social science",
    "social science": "social science",
    "hindi": "hindi",
}


def canonical_subject(s: str | None) -> str:
    """Lowercase, collapse whitespace, and map known aliases to a canonical subject."""
    key = " ".join((s or "").strip().lower().split())
    return _ALIASES.get(key, key)


def subject_matches(query_subject: str | None, record_subject: str | None) -> bool:
    """True when a (possibly shorthand) query subject refers to the record's subject.

    An empty query subject matches everything (no filter). Otherwise we match on canonical
    equality or a shared canonical word, so "Maths" == "Mathematics" and "Science" overlaps
    "Social Science", while "English" never matches "Mathematics".
    """
    if not query_subject:
        return True
    cq = canonical_subject(query_subject)
    cr = canonical_subject(record_subject)
    if not cr:
        return False
    if cq == cr:
        return True
    return bool(set(cq.split()) & set(cr.split()))
