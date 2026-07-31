from abc import ABC
from dataclasses import dataclass

from src.core.state import ClassicGameState


class Percept(ABC):
    pass


@dataclass
class ClassicPercept(Percept):
    state: ClassicGameState
    my_player_id: int
