from abc import ABC, abstractmethod
from typing import Optional


class Node(ABC):
    """Base class for all pipeline nodes."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._next: Optional[Node] = None

    def connect(self, other: "Node") -> "Node":
        """Chain this node's output to another node. Returns other for fluent API."""
        self._next = other
        return other

    def __rshift__(self, other: "Node") -> "Node":
        """Syntactic sugar: node_a >> node_b"""
        return self.connect(other)

    @abstractmethod
    def process(self, data: dict) -> dict:
        """Process incoming data and return modified/enriched data dict."""
        ...

    def validate(self, data: dict) -> None:
        """Optional: check that required input fields exist before processing."""
        pass
