from __future__ import annotations

import json
import sys
from typing import Callable, Protocol

from .models import EngineEvent


class EventSink(Protocol):
    def __call__(self, event: EngineEvent) -> None:
        ...


def event_to_json(event: EngineEvent) -> str:
    return json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)


def stdout_event_sink(event: EngineEvent) -> None:
    sys.stdout.write(event_to_json(event) + "\n")
    sys.stdout.flush()


def collect_events() -> tuple[list[dict], EventSink]:
    events: list[dict] = []

    def sink(event: EngineEvent) -> None:
        events.append(event.to_dict())

    return events, sink
