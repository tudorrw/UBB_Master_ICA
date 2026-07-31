from src.core.environment import Environment
from src.core.state import ClassicGameState
from src import keys


class ClassicTicTacToeEnvironment(Environment):
    def __init__(self) -> None:
        self.state = ClassicGameState()

    def current_state(self) -> ClassicGameState:
        return self.state

    def set_initial_state(self, state: ClassicGameState) -> None:
        self.state = state

    def get_grid(self) -> list[list[int]]:
        return self.state.grid

    def is_valid_move(self, row: int, col: int, player: int) -> bool:
        return self.state.is_valid_move(row, col)

    def apply_move(self, row: int, col: int, player: int) -> None:
        self.state.mark(row, col, player)

    def check_outcome(self) -> dict:
        return {keys.GAME_OVER: self.state.game_over, keys.WINNER: self.state.winner}
