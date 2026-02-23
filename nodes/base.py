"""
This is the basic node interface from which all of the nodes inherit.

It consists of main methods for creating a node, channeling nodes, processing and validating data.
"""
from abc import ABC, abstractmethod
from typing import Optional


class Node(ABC):
    """Base class for all pipeline nodes."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._next: Optional[Node] = None

    def then(self, other: "Node") -> "Node":
        # Register `other` as the node to run after this one finishes.
        # Returning `other` lets us chain calls: a.then(b).then(c).then(d)
        # — each .then() call wires up the next link and passes it forward.
        self._next = other
        return other

    def run(self, data: dict) -> dict:
        # This is where "inner calling" happens: each node is responsible for
        # handing off to the next one rather than an external loop doing it.
        data = self.process(data)

        if self._next is not None:
            # Pass the enriched data dict down the chain.
            # Because this is a recursive call, the last node in the chain
            # is the one that eventually prints "Done." and returns.
            return self._next.run(data)
        return data

    @abstractmethod
    def process(self, data: dict) -> dict:
        """Process incoming data and return modified/enriched data dict."""
        pass

