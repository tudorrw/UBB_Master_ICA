from abc import ABC, abstractmethod

from src.core.action import PlaceMarkAction
from src.core.percept import Percept


class Agent(ABC):
    @abstractmethod
    def see(self, percept: Percept) -> None:
        pass

    @abstractmethod
    def action(self) -> PlaceMarkAction | None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass
