"""Per-student durable memory.

Everything is namespaced under ``memory/<student_id>/`` and fetched by the active
student id only — the store for one student never has another student's data in
scope (structural half of the sec 8.4 isolation guarantee).
"""

import json
import shutil
from pathlib import Path
from typing import Any

from src.utils.config import load_config, repo_path
from src.utils.logging_config import get_logger

log = get_logger("memory_store")


class MemoryStore:
    def __init__(self, student_id: str, config: dict | None = None) -> None:
        config = config or load_config()
        mem = config.get("memory", {})
        self.student_id = student_id
        self.dir = repo_path(mem.get("root", "memory")) / student_id
        self.context_log_name = config.get("context", {}).get("log_file", "context_log.jsonl")

    def _path(self, name: str) -> Path:
        return self.dir / name

    def _ensure(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- mastery ----------------------------------------------------------
    def load_mastery(self) -> dict[str, Any]:
        p = self._path("mastery_scores.json")
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save_mastery(self, mastery: dict[str, Any]) -> None:
        self._ensure()
        self._path("mastery_scores.json").write_text(
            json.dumps(mastery, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- persona ----------------------------------------------------------
    def load_persona(self) -> dict[str, Any]:
        p = self._path("persona_overrides.json")
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save_persona(self, persona: dict[str, Any]) -> None:
        self._ensure()
        self._path("persona_overrides.json").write_text(
            json.dumps(persona, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- episodic log (append-only JSONL) ---------------------------------
    def append_episode(self, entry: dict[str, Any]) -> None:
        self._ensure()
        with open(self._path("episodic_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_episodes(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self._path("episodic_log.jsonl"))

    def read_session_episodes(self, session_id: str) -> list[dict[str, Any]]:
        return [e for e in self.read_episodes() if e.get("session_id") == session_id]

    # ---- context debug log (sec 14.4 audit reads this) --------------------
    def append_context_log(self, entry: dict[str, Any]) -> None:
        self._ensure()
        with open(self._path(self.context_log_name), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_context_logs(self) -> list[dict[str, Any]]:
        return self._read_jsonl(self._path(self.context_log_name))

    # ---- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        if self.dir.exists():
            shutil.rmtree(self.dir)
        log.info("cleared memory for %s", self.student_id)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
