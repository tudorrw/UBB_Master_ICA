import numpy as np
from src.constants import ROWS, COLS


class TicTacToeBoard:
    def __init__(self):
        self.squares = np.zeros((ROWS, COLS))

    def _find_winner(self):
        for col in range(COLS):  #check for vertical wins for each player
            for player in [1, 2]:
                if np.all(self.squares[:, col] == player):
                    return player, 'vertical', col

        for row in range(ROWS): #check for horizontal wins for each player
            for player in [1, 2]:
                if np.all(self.squares[row, :] == player):
                    return player, 'horizontal', row

        for player in [1, 2]: #check for main diagonal wins for each player
            if np.all(np.diag(self.squares) == player):
                return player, 'main_diag', None

        for player in [1, 2]: #check for anti-diagonal wins for each player
            if np.all(np.diag(np.fliplr(self.squares)) == player):
                return player, 'anti_diag', None

        return 0, None, None

    def final_state(self):
        '''
            @return 0 if there is no win yet
            @return 1 if player 1 wins
            @return 2 if player 2 wins
        '''
        return self._find_winner()[0]

    def get_win_line(self):
        player, won_type, index = self._find_winner()
        if player == 0:
            return None
        return player, won_type, index

    def mark_square(self, row, col, player):
        self.squares[row][col] = player

    def empty_square(self, row, col):
        return self.squares[row][col] == 0

    def get_empty_squares(self):
        empty_squares = []
        for row in range(ROWS):
            for col in range(COLS):
                if self.empty_square(row, col):
                    empty_squares.append((row, col))
        return empty_squares

    def is_full(self):
        return self.squares.all() != 0
