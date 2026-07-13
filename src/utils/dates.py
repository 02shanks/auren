"""Tolerant date parsing shared by tools and the mastery engine."""

import datetime as dt

_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%b %d, %Y", "%d %b %Y")


def parse_date(value: str | None) -> dt.date | None:
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    for fmt in _FORMATS:
        try:
            return dt.datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    try:
        return dt.date.fromisoformat(v)
    except ValueError:
        return None


def days_until(target: dt.date, today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return (target - today).days
