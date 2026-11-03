"""Abstract base class every tab inherits from."""
from abc import ABC, abstractmethod


class BaseTab(ABC):
    """A tab has a label and knows how to render itself."""

    label: str = ""

    @abstractmethod
    def render(self) -> None:
        ...