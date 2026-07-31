import time

from src import keys
from src.core.action import PlaceMarkAction
from src.core.communication_agent import CommunicationAgent
from src.core.percept import Percept
from src.core.state import ClassicGameState
from src.classic.environment import ClassicTicTacToeEnvironment


class GameMasterAgent(CommunicationAgent):
    # WAIT_TIME = 1.5
    WAIT_TIME = 1

    def __init__(self, blackboard, environment: ClassicTicTacToeEnvironment) -> None:
        super().__init__(blackboard, f"{keys.GAME_MASTER}")
        self.environment = environment
        self._initialized = False

    def see(self, percept: Percept) -> None:
        pass

    def action(self) -> PlaceMarkAction | None:
        return None

    def run(self) -> None:
        while True:
            # initialization: set the environment's initial state and publish it
            if not self._initialized:
                self.environment.set_initial_state(ClassicGameState())
                self.write(keys.PENDING_MOVE, None)
                self._publish_state()
                self._initialized = True

            state = self.environment.current_state()
            if state.game_over:
                time.sleep(self.WAIT_TIME)
                continue

            pending_move = self.read(keys.PENDING_MOVE)
            # time.sleep(self.WAIT_TIME)
            if pending_move is not None:
                row, col = pending_move
                current_turn = state.current_turn

                if self.environment.is_valid_move(row, col, current_turn):
                    self.environment.apply_move(row, col, current_turn)
                    self.write(keys.PENDING_MOVE, None)
                    self._publish_state()
                else:
                    self.write(keys.PENDING_MOVE, None)
            time.sleep(self.WAIT_TIME)

    def _publish_state(self) -> None:
        state = self.environment.current_state()
        self.write(keys.BOARD, state.grid)
        self.write(keys.CURRENT_TURN, state.current_turn)
        self.write(keys.GAME_OVER, state.game_over)
        self.write(keys.WINNER, state.winner)
