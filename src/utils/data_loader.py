"""Defensive data loading + integrity validation for Auren.

Loads the given 4-file sample dataset and the generated multi-student synthetic
dataset behind one repository. Malformed records are sanitized-with-warning
rather than crashed on; a genuinely ambiguous integrity problem (duplicate student_id)
is surfaced as ``DataIntegrityError`` because silently merging two records
claiming the same id would be a data bug.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.utils.config import repo_path
from src.utils.logging_config import get_logger

log = get_logger("data_loader")

MAX_REASONABLE_STR = 200  # topic/title/name longer than this -> integrity warning (not fatal)


# --------------------------------------------------------------------------- #
# Domain models (lenient: extra keys ignored so injected junk cannot crash us) #
# --------------------------------------------------------------------------- #
class SubjectPerformance(BaseModel):
    model_config = ConfigDict(extra="ignore")
    subject: str
    overall_score_percentage: float | None = None
    trend: str | None = None


class Material(BaseModel):
    model_config = ConfigDict(extra="ignore")
    material_id: str
    topic: str
    title: str
    subject: str | None = None
    material_type: str | None = None


class UpcomingTest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    test_id: str
    subject: str = ""
    test_name: str = ""
    date: str | None = None  # raw string; parsed with error-handling in the tool
    topics: list[str] = Field(default_factory=list)


class StudentProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")
    student_id: str
    name: str = ""
    grade: int | None = None
    board: str | None = None
    target_exam: str | None = None
    daily_study_time_minutes: int | None = None
    strong_topics: list[str] = Field(default_factory=list)
    weak_topics: list[str] = Field(default_factory=list)


class StudentRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    profile: StudentProfile
    performance: list[SubjectPerformance] = Field(default_factory=list)
    tests: list[UpcomingTest] = Field(default_factory=list)

    @property
    def student_id(self) -> str:
        return self.profile.student_id


# --------------------------------------------------------------------------- #
# Integrity reporting                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class IntegrityIssue:
    student_id: str
    field: str
    message: str
    severity: str = "warning"  # "warning" | "error"


class DataIntegrityError(Exception):
    """Raised for unrecoverable integrity problems (e.g. a duplicated student_id)."""


# --------------------------------------------------------------------------- #
# Sanitizers                                                                   #
# --------------------------------------------------------------------------- #
def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x).strip() != ""]
    return []


def sanitize_profile(raw: dict[str, Any]) -> tuple[dict[str, Any], list[IntegrityIssue]]:
    issues: list[IntegrityIssue] = []
    sid = str(raw.get("student_id") or "").strip()
    name = "" if raw.get("name") is None else str(raw.get("name")).strip()
    grade = raw.get("grade") if isinstance(raw.get("grade"), int) else None
    dsm = raw.get("daily_study_time_minutes")
    clean = {
        "student_id": sid,
        "name": name,
        "grade": grade,
        "board": raw.get("board"),
        "target_exam": raw.get("target_exam"),
        "daily_study_time_minutes": dsm if isinstance(dsm, int) else None,
        "strong_topics": _as_str_list(raw.get("strong_topics")),
        "weak_topics": _as_str_list(raw.get("weak_topics")),
    }
    if not sid:
        issues.append(
            IntegrityIssue("<unknown>", "student_id", "missing/empty student_id", "error")
        )
    if name == "":
        issues.append(IntegrityIssue(sid, "name", "empty student name", "warning"))
    if raw.get("weak_topics") is None and "weak_topics" in raw:
        issues.append(IntegrityIssue(sid, "weak_topics", "null coerced to empty list", "warning"))
    if dsm is None:
        issues.append(
            IntegrityIssue(sid, "daily_study_time_minutes", "optional field absent/null", "warning")
        )
    # Contradiction: a topic in both strong and weak -> weak wins (safer than under-recommending).
    overlap = sorted(set(clean["strong_topics"]) & set(clean["weak_topics"]))
    for t in overlap:
        clean["strong_topics"].remove(t)
        issues.append(
            IntegrityIssue(
                sid,
                "strong_topics",
                f"topic '{t[:40]}' listed as both strong & weak; kept as weak (tie-break)",
                "warning",
            )
        )
    for t in clean["strong_topics"] + clean["weak_topics"]:
        if len(t) > MAX_REASONABLE_STR:
            issues.append(
                IntegrityIssue(
                    sid, "topic", f"topic string exceeds {MAX_REASONABLE_STR} chars", "warning"
                )
            )
            break
    if len(name) > MAX_REASONABLE_STR:
        issues.append(IntegrityIssue(sid, "name", "name exceeds length bound", "warning"))
    return clean, issues


def sanitize_performance(
    raw_list: Any, sid: str
) -> tuple[list[dict[str, Any]], list[IntegrityIssue]]:
    issues: list[IntegrityIssue] = []
    out: list[dict[str, Any]] = []
    if not isinstance(raw_list, list):
        if raw_list is not None:
            issues.append(
                IntegrityIssue(sid, "subject_performance", "expected list; ignored", "warning")
            )
        return out, issues
    for row in raw_list:
        if not isinstance(row, dict):
            continue
        subj = row.get("subject")
        if not subj:
            continue
        score = row.get("overall_score_percentage")
        if isinstance(score, int | float) and not isinstance(score, bool):
            if score < 0 or score > 100:
                issues.append(
                    IntegrityIssue(
                        sid,
                        "overall_score_percentage",
                        f"score {score} out of [0,100] for {subj}",
                        "warning",
                    )
                )
        else:
            score = None
        out.append(
            {"subject": str(subj), "overall_score_percentage": score, "trend": row.get("trend")}
        )
    return out, issues


def sanitize_tests(raw_list: Any, sid: str) -> tuple[list[dict[str, Any]], list[IntegrityIssue]]:
    issues: list[IntegrityIssue] = []
    out: list[dict[str, Any]] = []
    if not isinstance(raw_list, list):
        if raw_list is not None:
            issues.append(
                IntegrityIssue(sid, "upcoming_tests", "expected list; ignored", "warning")
            )
        return out, issues
    for row in raw_list:
        if not isinstance(row, dict):
            continue
        tid = row.get("test_id")
        if not tid:
            continue
        date = row.get("date")
        out.append(
            {
                "test_id": str(tid),
                "subject": str(row.get("subject") or ""),
                "test_name": str(row.get("test_name") or ""),
                "date": str(date) if date is not None else None,
                "topics": _as_str_list(row.get("topics")),
            }
        )
    return out, issues


def sanitize_material(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    mid = raw.get("material_id")
    topic = raw.get("topic")
    title = raw.get("title")
    if not mid or (not topic and not title):
        return None
    return {
        "material_id": str(mid),
        "topic": str(topic or title),
        "title": str(title or topic),
        "subject": raw.get("subject"),
        "material_type": raw.get("material_type"),
    }


# --------------------------------------------------------------------------- #
# Repository                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class DatasetRepository:
    students: dict[str, StudentRecord] = field(default_factory=dict)
    materials_list: list[Material] = field(default_factory=list)
    issues: list[IntegrityIssue] = field(default_factory=list)
    duplicate_ids: set[str] = field(default_factory=set)

    def canonical_id(self, student_id: str) -> str:
        """Resolve a student id to its stored form, tolerant of surrounding whitespace and case.

        Returns the stripped input unchanged when there is no match, so not-found paths still work.
        """
        s = (student_id or "").strip()
        if s in self.students or s in self.duplicate_ids:
            return s
        low = s.casefold()
        for actual in (*self.students, *self.duplicate_ids):
            if actual.casefold() == low:
                return actual
        return s

    def get_student(self, student_id: str) -> StudentRecord | None:
        student_id = self.canonical_id(student_id)
        if student_id in self.duplicate_ids:
            raise DataIntegrityError(
                f"student_id '{student_id}' appears in multiple records; "
                "refusing to guess which is authoritative"
            )
        return self.students.get(student_id)

    def student_ids(self) -> list[str]:
        return sorted(self.students)

    def materials(self) -> list[Material]:
        return list(self.materials_list)

    def integrity_report(self) -> list[IntegrityIssue]:
        return list(self.issues)

    def errors(self) -> list[IntegrityIssue]:
        return [i for i in self.issues if i.severity == "error"]


def _read_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        log.warning("malformed JSON in %s: %s", path, exc)
        return None


def _add_student(
    repo: DatasetRepository, profile_raw: dict[str, Any], perf_raw: Any, tests_raw: Any
) -> None:
    prof_clean, issues = sanitize_profile(profile_raw)
    repo.issues.extend(issues)
    sid = prof_clean["student_id"]
    if not sid:
        return
    perf_clean, perf_issues = sanitize_performance(perf_raw, sid)
    tests_clean, test_issues = sanitize_tests(tests_raw, sid)
    repo.issues.extend(perf_issues)
    repo.issues.extend(test_issues)
    record = StudentRecord(
        profile=StudentProfile(**prof_clean),
        performance=[SubjectPerformance(**p) for p in perf_clean],
        tests=[UpcomingTest(**t) for t in tests_clean],
    )
    if sid in repo.students:
        repo.duplicate_ids.add(sid)
        repo.issues.append(
            IntegrityIssue(sid, "student_id", "duplicate student_id across records", "error")
        )
    else:
        repo.students[sid] = record


def _load_materials(repo: DatasetRepository, mats_raw: list[Any]) -> None:
    seen = {m.material_id for m in repo.materials_list}
    for raw in mats_raw:
        clean = sanitize_material(raw)
        if not clean or clean["material_id"] in seen:
            continue
        repo.materials_list.append(Material(**clean))
        seen.add(clean["material_id"])


def load_sample(base: Path | None = None) -> DatasetRepository:
    base = base or repo_path("data")
    repo = DatasetRepository()
    profile = _read_json(base / "student_profile.json") or {}
    perf = (_read_json(base / "performance_history.json") or {}).get("subject_performance")
    tests = (_read_json(base / "upcoming_tests.json") or {}).get("upcoming_tests")
    _add_student(repo, profile, perf, tests)
    mats = (_read_json(base / "study_materials.json") or {}).get("materials") or []
    _load_materials(repo, mats)
    return repo


def load_synthetic(base: Path | None = None) -> DatasetRepository:
    base = base or repo_path("data", "synthetic")
    repo = DatasetRepository()
    sdir = base / "students"
    if sdir.exists():
        for fp in sorted(sdir.glob("*.json")):
            bundle = _read_json(fp) or {}
            _add_student(
                repo, bundle.get("profile") or {}, bundle.get("performance"), bundle.get("tests")
            )
    mats = (_read_json(base / "study_materials_extended.json") or {}).get("materials") or []
    _load_materials(repo, mats)
    return repo


def load_dataset(name: str = "sample") -> DatasetRepository:
    if name == "sample":
        return load_sample()
    if name == "synthetic":
        return load_synthetic()
    if name == "all":
        repo = load_sample()
        syn = load_synthetic()
        for sid, rec in syn.students.items():
            if sid in repo.students:
                repo.duplicate_ids.add(sid)
            else:
                repo.students[sid] = rec
        repo.duplicate_ids |= syn.duplicate_ids
        _load_materials(repo, [m.model_dump() for m in syn.materials_list])
        repo.issues.extend(syn.issues)
        return repo
    raise ValueError(f"unknown dataset '{name}'")


def load_injection_materials(base: Path | None = None) -> list[Material]:
    """Load the deliberately poisoned material fixtures (kept separate from clean data)."""
    base = base or repo_path("data", "synthetic")
    raw = (_read_json(base / "injection_fixtures.json") or {}).get("materials") or []
    out: list[Material] = []
    for row in raw:
        clean = sanitize_material(row)
        if clean:
            out.append(Material(**clean))
    return out
