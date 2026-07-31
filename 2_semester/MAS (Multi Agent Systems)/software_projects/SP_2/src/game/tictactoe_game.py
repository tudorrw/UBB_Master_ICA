import numpy as np

from src.constants import COLS, ROWS
from src.game.tictactoe_board import TicTacToeBoard
from src.game.tictactoe_renderer import TicTacToeRenderer


class TicTacToeGame:
    def __init__(self, screen) -> None:
        self.screen = screen
        self.board = TicTacToeBoard()
        self.renderer = TicTacToeRenderer(screen)
        self.renderer.draw_lines()

    def sync_from_grid(self, grid: list[list[int]]) -> None:
        self.board.squares = np.array(grid, dtype=int)
        self.renderer.draw_lines()
        for row in range(ROWS):
            for col in range(COLS):
                player = int(grid[row][col])
                if player != 0:
                    self.renderer.draw_figure(row, col, player)
        self.is_over()

    def is_over(self) -> bool:
        win_line = self.board.get_win_line()
        if win_line:
            player, won_type, index = win_line
            self.renderer.draw_winner_line(player, won_type, index)
            return True
        return self.board.is_full()
