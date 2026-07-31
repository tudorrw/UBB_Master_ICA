from abc import ABC, abstractmethod

from src.core.state import State


class Environment(ABC):
    state: State

    @abstractmethod
    def current_state(self) -> State:
        pass

    @abstractmethod
    def set_initial_state(self, state: State) -> None:
        pass

    @abstractmethod
    def get_grid(self) -> list[list[int]]:
        pass

    @abstractmethod
    def is_valid_move(self, row: int, col: int, player: int) -> bool:
        pass

    @abstractmethod
    def apply_move(self, row: int, col: int, player: int) -> None:
        pass

    @abstractmethod
    def check_outcome(self) -> dict:
        pass
