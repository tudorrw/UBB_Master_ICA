from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PlaceMarkAction:
    row: int
    col: int


class Action(ABC):
    @abstractmethod
    def execute(self) -> None:
        pass
