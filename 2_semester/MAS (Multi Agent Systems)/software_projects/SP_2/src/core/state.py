from abc import ABC, abstractmethod

import numpy as np

from src.game.tictactoe_board import TicTacToeBoard


class State(ABC):
    @abstractmethod
    def display(self) -> str:
        pass


class ClassicGameState(State):
    def __init__(
        self,
        grid: list[list[int]] | None = None,
        current_turn: int = 1,
        game_over: bool = False,
        winner: int = 0,
    ) -> None:
        self.board = TicTacToeBoard()
        if grid is not None:
            self.board.squares = np.array(grid, dtype=int)
        self.current_turn = current_turn
        self.game_over = game_over
        self.winner = winner

    @property
    def grid(self) -> list[list[int]]:
        return self.board.squares.astype(int).tolist()

    def is_valid_move(self, row: int, col: int) -> bool:
        if self.board.final_state() != 0:
            return False
        return self.board.empty_square(row, col)

    def mark(self, row: int, col: int, player: int) -> None:
        self.board.mark_square(row, col, player)
        self._refresh_outcome()
        if not self.game_over:
            self.current_turn = 1 if player == 2 else 2

    def _refresh_outcome(self) -> None:
        winner = self.board.final_state()
        if winner != 0:
            self.game_over = True
            self.winner = winner
        elif self.board.is_full():
            self.game_over = True
            self.winner = 0
        else:
            self.game_over = False
            self.winner = 0

    def display(self) -> str:
        symbols = {0: " ", 1: "X", 2: "O"}
        lines = []
        for i, row in enumerate(self.grid):
            cells = [symbols[int(cell)] for cell in row]
            lines.append(f" {cells[0]} | {cells[1]} | {cells[2]} ")
            if i < len(self.grid) - 1:
                lines.append("---+---+---")
        return "\n".join(lines)
