"""Shared execution context handed to every tool (keeps tools decoupled from the orchestrator)."""

import datetime as dt
from dataclasses import dataclass

from src.memory.mastery import MasteryEngine
from src.retrieval.indexer import MaterialRetriever
from src.utils.data_loader import DatasetRepository


@dataclass
class ToolContext:
    repo: DatasetRepository
    retriever: MaterialRetriever
    mastery: MasteryEngine
    config: dict
    today: dt.date | None = None

    def now(self) -> dt.date:
        return self.today or dt.date.today()
