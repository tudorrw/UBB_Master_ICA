import copy

from src import keys
from src.core.action import PlaceMarkAction
from src.agents.strategic_agent import StrategicAgent
from src.classic.environment import ClassicTicTacToeEnvironment
from src.game.tictactoe_board import TicTacToeBoard


class MinimaxStrategicAgent(StrategicAgent):
    def __init__(
        self,
        blackboard,
        player_id: int,
        environment: ClassicTicTacToeEnvironment,
        max_depth: int = 7,
    ) -> None:
        super().__init__(blackboard, player_id, environment, f"{keys.MINIMAX_AGENT}: Player{player_id}")
        self.max_depth = max_depth

    def action(self) -> PlaceMarkAction | None:
        if self._percept is None:
            return None

        board = copy.deepcopy(self._percept.state.board)
        maximizing = self.player_id == 1
        _, move = self._minimax(board, self.max_depth, maximizing, float("-inf"), float("inf"))

        if move is None:
            return None

        row, col = move
        return PlaceMarkAction(row=row, col=col)

    def _score(self, board: TicTacToeBoard) -> int:
        winner = board.final_state()
        if winner == 1:
            return 1
        if winner == 2:
            return -1
        return 0

    def _minimax(self, board: TicTacToeBoard, depth: int, maximizing: bool, alpha: float, beta: float
        )  -> tuple[int, tuple[int, int] | None]:
        
        if board.final_state() != 0 or board.is_full() or depth == 0:
            return self._score(board), None

        empty_squares = board.get_empty_squares()

        if maximizing:
            best_eval = float("-inf")
            best_move = None
            for row, col in empty_squares:
                temp_board = copy.deepcopy(board)
                temp_board.mark_square(row, col, 1)
                eval_score, _ = self._minimax(temp_board, depth - 1, False, alpha, beta)
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = (row, col)
                alpha = max(alpha, best_eval)
                if beta <= alpha:
                    break
            return best_eval, best_move

        best_eval = float("inf")
        best_move = None
        for row, col in empty_squares:
            temp_board = copy.deepcopy(board)
            temp_board.mark_square(row, col, 2)
            eval_score, _ = self._minimax(temp_board, depth - 1, True, alpha, beta)
            if eval_score < best_eval:
                best_eval = eval_score
                best_move = (row, col)
            beta = min(beta, best_eval)
            if beta <= alpha:
                break
        return best_eval, best_move
