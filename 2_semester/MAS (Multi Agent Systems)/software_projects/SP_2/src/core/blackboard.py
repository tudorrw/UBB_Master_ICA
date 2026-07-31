import multiprocessing
from queue import Empty


class Blackboard:
    def __init__(self) -> None:
        manager = multiprocessing.Manager()
        self._store = manager.dict()
        self._queue = manager.Queue()
        self._lock = manager.Lock()

    def read(self, key: str):
        with self._lock:
            return self._store.get(key)

    def write(self, key: str, value, source: str = "Blackboard") -> None:
        with self._lock:
            self._store[key] = value
        self._queue.put(f"{source} wrote {key}: {value}")

    def pop_message(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None
