"""Deterministic mastery-priority scoring.

    priority_score(t) = clamp01(
          w_weak     * weakness_signal(t)
        + w_urgency  * test_urgency(t)
        + w_stale    * staleness(t)
        - w_feedback * positive_feedback_decay(t) )

The score is fully deterministic and interpretable (every component is stored),
so a ``log_feedback(signal="helped")`` event produces a visibly different ranking
next time — this is the concrete self-improvement signal, not an ever-growing prompt.
"""

import datetime as dt

from src.utils.data_loader import StudentRecord
from src.utils.dates import parse_date


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


class MasteryEngine:
    def __init__(self, config: dict) -> None:
        m = config.get("mastery", {})
        self.w_weak = float(m.get("w_weakness", 0.40))
        self.w_urgency = float(m.get("w_test_urgency", 0.30))
        self.w_stale = float(m.get("w_staleness", 0.20))
        self.w_feedback = float(m.get("w_positive_feedback", 0.20))
        self.near = int(m.get("test_urgency_near_days", 3))
        self.far = int(m.get("test_urgency_far_days", 21))
        self.stale_full = int(m.get("staleness_full_days", 14))
        self.fb_halflife = 7.0

    # ---- component signals ------------------------------------------------
    @staticmethod
    def _topics(rec: StudentRecord) -> set[str]:
        topics = set(rec.profile.weak_topics) | set(rec.profile.strong_topics)
        for test in rec.tests:
            topics |= set(test.topics)
        return topics

    @staticmethod
    def _topic_subject(rec: StudentRecord) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for test in rec.tests:
            for topic in test.topics:
                mapping.setdefault(topic, test.subject)
        return mapping

    @staticmethod
    def _subject_score(rec: StudentRecord) -> dict[str, float]:
        return {
            p.subject: p.overall_score_percentage
            for p in rec.performance
            if p.overall_score_percentage is not None
        }

    def weakness_signal(
        self, t: str, rec: StudentRecord, subj_of: dict[str, str], scores: dict[str, float]
    ) -> float:
        if t in set(rec.profile.weak_topics):
            return 1.0
        if t in set(rec.profile.strong_topics):
            return 0.1
        subj = subj_of.get(t)
        s = scores.get(subj) if subj else None
        if s is None:
            return 0.5
        return clamp01((100.0 - s) / 100.0)

    def test_urgency(self, t: str, rec: StudentRecord, today: dt.date) -> float:
        best: int | None = None
        for test in rec.tests:
            if t not in test.topics:
                continue
            d = parse_date(test.date)
            if d is None:
                continue
            du = (d - today).days
            if du < 0:
                continue
            best = du if best is None else min(best, du)
        if best is None:
            return 0.0
        if best <= self.near:
            return 1.0
        if best >= self.far:
            return 0.0
        return (self.far - best) / (self.far - self.near)

    def staleness(self, hist: dict, today: dt.date) -> float:
        ls = parse_date(hist.get("last_studied")) or parse_date(hist.get("last_recommended"))
        if ls is None:
            return 1.0  # never touched -> treated as overdue
        days = max(0, (today - ls).days)
        return clamp01(days / self.stale_full)

    def positive_feedback_decay(self, hist: dict, today: dt.date) -> float:
        pf = float(hist.get("positive_feedback", 0.0))
        if pf <= 0.0:
            return 0.0
        updated = parse_date(hist.get("positive_feedback_updated"))
        if updated is None:
            return pf
        days = max(0, (today - updated).days)
        return pf * (0.5 ** (days / self.fb_halflife))

    # ---- public API -------------------------------------------------------
    def recompute(
        self, rec: StudentRecord, prior: dict | None = None, today: dt.date | None = None
    ) -> dict:
        today = today or dt.date.today()
        prior = prior or {}
        subj_of = self._topic_subject(rec)
        scores = self._subject_score(rec)
        topics = self._topics(rec) | set(prior)
        out: dict[str, dict] = {}
        for t in sorted(topics):
            hist = prior.get(t, {})
            weak = self.weakness_signal(t, rec, subj_of, scores)
            urg = self.test_urgency(t, rec, today)
            stale = self.staleness(hist, today)
            pfd = self.positive_feedback_decay(hist, today)
            raw = (
                self.w_weak * weak
                + self.w_urgency * urg
                + self.w_stale * stale
                - self.w_feedback * pfd
            )
            out[t] = {
                "priority_score": round(clamp01(raw), 4),
                "components": {
                    "weakness": round(weak, 4),
                    "test_urgency": round(urg, 4),
                    "staleness": round(stale, 4),
                    "positive_feedback_decay": round(pfd, 4),
                },
                "feedback_count": int(hist.get("feedback_count", 0)),
                "positive_feedback": round(float(hist.get("positive_feedback", 0.0)), 4),
                "positive_feedback_updated": hist.get("positive_feedback_updated"),
                "last_studied": hist.get("last_studied"),
                "last_recommended": hist.get("last_recommended"),
                "last_updated": today.isoformat(),
            }
        return out

    def apply_feedback(
        self, prior: dict, topic: str, signal: str, today: dt.date | None = None
    ) -> dict:
        today = today or dt.date.today()
        rec = prior.setdefault(topic, {"feedback_count": 0, "positive_feedback": 0.0})
        rec["feedback_count"] = int(rec.get("feedback_count", 0)) + 1
        pf = float(rec.get("positive_feedback", 0.0))
        if signal in ("helped", "positive", "up", "good"):
            pf = min(1.0, pf + 0.6)
        elif signal in ("not_helped", "negative", "down", "bad"):
            pf = max(0.0, pf - 0.3)
        rec["positive_feedback"] = pf
        rec["positive_feedback_updated"] = today.isoformat()
        rec["last_studied"] = today.isoformat()
        return prior

    def mark_recommended(self, prior: dict, topic: str, today: dt.date | None = None) -> dict:
        today = today or dt.date.today()
        rec = prior.setdefault(topic, {"feedback_count": 0, "positive_feedback": 0.0})
        rec["last_recommended"] = today.isoformat()
        return prior

    @staticmethod
    def ranked_topics(mastery: dict) -> list[tuple[str, float]]:
        pairs = [(t, v.get("priority_score", 0.0)) for t, v in mastery.items()]
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs
