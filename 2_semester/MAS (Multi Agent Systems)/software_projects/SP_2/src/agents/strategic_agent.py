import time

from src import keys
from src.core.action import PlaceMarkAction
from src.core.communication_agent import CommunicationAgent
from src.core.percept import ClassicPercept, Percept
from src.core.state import ClassicGameState
from src.classic.environment import ClassicTicTacToeEnvironment


class StrategicAgent(CommunicationAgent):
    WAIT_TIME = 2

    def __init__(self, blackboard, player_id: int, environment: ClassicTicTacToeEnvironment, name: str) -> None:
        super().__init__(blackboard, name)
        self.player_id = player_id
        self.environment = environment
        self._percept: ClassicPercept | None = None

    def see(self, percept: Percept) -> None:
        self._percept = percept

    def run(self) -> None:
        while True:
            game_over = self.read(keys.GAME_OVER)
            if game_over:
                time.sleep(self.WAIT_TIME)
                continue

            current_turn = self.read(keys.CURRENT_TURN)
            pending_move = self.read(keys.PENDING_MOVE)
            grid = self.read(keys.BOARD)

            if (grid is not None and current_turn == self.player_id and pending_move is None):
                state = ClassicGameState(
                    grid=grid,
                    current_turn=current_turn,
                    game_over=bool(game_over),
                    winner=self.read(keys.WINNER) or 0,
                )
                percept = ClassicPercept(state=state, my_player_id=self.player_id)
                self.see(percept)
                move = self.action()
                if move is not None:
                    self.write(keys.PENDING_MOVE, (move.row, move.col))

            time.sleep(self.WAIT_TIME)
