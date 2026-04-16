from __future__ import annotations
from multiprocessing import Queue
from queue import Empty, Full
from typing import Any


def put_latest(queue_obj: Queue[Any], item: Any) -> None:
    try:
        queue_obj.put(item, block=False)
    except Full:
        try:
            queue_obj.get(block=False)
        except Empty:
            pass
        try:
            queue_obj.put(item, block=False)
        except Full:
            pass


def drain_latest(queue_obj: Queue[Any]) -> Any | None:
    latest = None
    while True:
        try:
            latest = queue_obj.get(block=False)
        except Empty:
            break
    return latest
