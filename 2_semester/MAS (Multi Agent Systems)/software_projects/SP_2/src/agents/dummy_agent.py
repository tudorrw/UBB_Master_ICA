import random

from src import keys
from src.core.action import PlaceMarkAction
from src.agents.strategic_agent import StrategicAgent
from src.classic.environment import ClassicTicTacToeEnvironment


class DummyStrategicAgent(StrategicAgent):
    def __init__(
        self,
        blackboard,
        player_id: int,
        environment: ClassicTicTacToeEnvironment,
    ) -> None:
        super().__init__(blackboard, player_id, environment, f"{keys.DUMMY_AGENT}: Player{player_id}")

    def action(self) -> PlaceMarkAction | None:
        if self._percept is None:
            return None

        grid = self._percept.state.grid
        empty = []
        for row in range(len(grid)):
            for col in range(len(grid[row])):
                if grid[row][col] == 0:
                    empty.append((row, col))

        if not empty:
            return None

        row, col = empty[random.randint(0, len(empty) - 1)]
        return PlaceMarkAction(row=row, col=col)
