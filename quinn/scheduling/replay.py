"""Replay consumes ordered per-candidate streams without returning future samples in quotes."""
from quinn.estimation import Observation
from quinn.scheduling import Quote


class Replay:
    def __init__(self, streams, bounds):
        self._streams = {key: tuple(Observation(**item) for item in stream)
                         for key, stream in streams.items()}
        self._bounds = bounds
        self._consumed = dict.fromkeys(streams, 0)

    def quote(self, candidate, index):
        key = candidate.key
        if index >= len(self._streams[key]):
            return None
        return Quote(*self._bounds[key])

    def acquire(self, candidate, index):
        key = candidate.key
        if index != self._consumed[key]:
            raise ValueError("replay only exposes the next observation")
        self._consumed[key] += 1
        return self._streams[key][index]
