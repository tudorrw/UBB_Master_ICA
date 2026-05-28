import sys
import pygame 

import numpy as np
import copy
from src.constants import *

pygame.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")

screen.fill(BG_COLOR)

class Board:
    def __init__(self):
        self.squares = np.zeros((ROWS, COLS))
        
    def final_state(self, show=False):
        '''
            @return 0 if there is no win yet
            @return 1 if player 1 wins
            @return 2 if player 2 wins
        '''
        # vertical wins
        for col in range(COLS):
            for player in [1, 2]:
                if np.all(self.squares[:, col] == player):
                    if show: self.draw_winner(player, 'vertical', col)
                    return player

        # horizontal wins
        for row in range(ROWS):
            for player in [1, 2]:
                if np.all(self.squares[row, :] == player):
                    if show: self.draw_winner(player, 'horizontal', row)
                    return player

        # main diagonal win ( \ )
        for player in [1, 2]:
            if np.all(np.diag(self.squares) == player):
                if show: self.draw_winner(player, 'main_diag', None)
                return player

        # anti diagonal win ( / )
        for player in [1, 2]:
            if np.all(np.diag(np.fliplr(self.squares)) == player):
                if show: self.draw_winner(player, 'anti_diag', None)
                return player

        return 0
        
    def draw_winner(self, player, won_type, index):
        color = CROSS_COLOR if player == 1 else CIRCLE_COLOR

        if won_type == 'vertical':
            start = (index * SQSIZE + SQSIZE // 2, 20)
            end   = (index * SQSIZE + SQSIZE // 2, HEIGHT - 20)

        elif won_type == 'horizontal':
            start = (20, index * SQSIZE + SQSIZE // 2)
            end   = (WIDTH - 20, index * SQSIZE + SQSIZE // 2)

        elif won_type == 'main_diag':
            start = (20, 20)
            end   = (WIDTH - 20, HEIGHT - 20)

        elif won_type == 'anti_diag':
            start = (WIDTH - 20, 20)
            end   = (20, HEIGHT - 20)

        pygame.draw.line(screen, color, start, end, LINE_WIDTH)
        
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
        # return list(zip(*np.where(self.squares == 0)))
    
    
    def is_full(self):
        return self.squares.all() != 0
    
    def is_empty(self):
        return self.squares.all() == 0
    
    
class AI:
    def __init__(self, level = 1, player = 2):
        self.level = level
        self.player = player
        
    
    def eval(self, main_board):  
        if(self.level == 0):
            # random choice
            eval = "random"
            move = self.randon_choice(main_board)
        else: 
            # minimax algorithm choice
            eval, move = self.minimax(main_board, False)
        print(f"Ai has chosen to mark the square in pos {move} with an eval of {eval}")
        return move #row, col
     
    def minimax(self, board, maximizing):
        case = board.final_state()
        if case == 1:
            return 1, None #eval, move
        elif case == 2:
            return -1, None
        elif board.is_full():
            return 0, None
        
        if maximizing:
            max_eval = float('-inf')
            best_move = None
            empty_squares = board.get_empty_squares()
            for (row, col) in empty_squares:
                temp_board = copy.deepcopy(board)
                temp_board.mark_square(row, col, 1)
                eval = self.minimax(temp_board, False)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = (row, col)
            return max_eval, best_move
        
        elif not maximizing:
            min_eval = float('inf')
            best_move = None
            empty_squares = board.get_empty_squares()
            
            for (row, col) in empty_squares:
                temp_board = copy.deepcopy(board)
                temp_board.mark_square(row, col, self.player)
                eval = self.minimax(temp_board, True)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = (row, col)
            return min_eval, best_move


    
    
    def randon_choice(self, board):
        empty_squares = board.get_empty_squares()
        index = np.random.randint(0, len(empty_squares))
        return empty_squares[index]

    
class Game:
    
    def __init__(self):
        self.board = Board()
        self.ai = AI()
        self.player = 1 # 1 for crosses, 2 for circles
        self.game_mode = 'ai' # pvp or ai
        self.running = True
        self.show_lines()
    
    def show_lines(self):
        screen.fill(BG_COLOR)
        #vertical lines
        pygame.draw.line(screen, LINE_COLOR, (SQSIZE, 0), (SQSIZE, HEIGHT), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (2*SQSIZE, 0), (2*SQSIZE, HEIGHT), LINE_WIDTH)
        
        #horizontal lines
        pygame.draw.line(screen, LINE_COLOR, (0, SQSIZE), (WIDTH, SQSIZE), LINE_WIDTH)
        pygame.draw.line(screen, LINE_COLOR, (0, 2*SQSIZE), (WIDTH, 2*SQSIZE), LINE_WIDTH)
    
    def draw_figure(self, row, col):
        if self.player == 1:
            #draw cross
            # descending line
            start_descending = (col * SQSIZE + OFFSET, row * SQSIZE + OFFSET)
            end_descending = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            pygame.draw.line(screen, CROSS_COLOR, start_descending, end_descending, CROSS_WIDTH)
            # ascending line
            start_ascending = (col * SQSIZE + OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            end_ascending = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + OFFSET)
            pygame.draw.line(screen, CROSS_COLOR, start_ascending, end_ascending, CROSS_WIDTH)
            
        elif self.player == 2:
            #draw circle
            center = (col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2)
            pygame.draw.circle(screen, CIRCLE_COLOR, center, CIRCLE_RADIUS, CIRCLE_WIDTH)
    
    def make_move(self, row, col):
        self.board.mark_square(row, col,self.player)
        self.draw_figure(row, col)
        self.next_turn()

    def next_turn(self):
        self.player = self.player % 2 + 1
        
    def change_game_mode(self):
        self.game_mode = 'ai' if self.game_mode == 'pvp' else 'pvp'
    
    def is_over(self):
        return self.board.final_state(show = True) != 0 or self.board.is_full()
    
    def reset(self):
        self.__init__()

        
def main():
    
    game = Game()
    ai = game.ai
    board = game.board
    
    while True:
        
        for event in pygame.event.get():
      
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_g:
                    game.change_game_mode()
                    
                if event.key == pygame.K_r:
                    game.reset()
                    ai = game.ai
                    board = game.board

                if event.key == pygame.K_0:
                    ai.level = 0
                
                if event.key == pygame.K_1:
                    ai.level = 1
            
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                row = pos[1] // SQSIZE
                col = pos[0] // SQSIZE
                
                
                if board.empty_square(row, col) and game.running:
                    game.make_move(row, col)
                
                    if game.is_over():
                        game.running = False
                    
            
        if game.game_mode == 'ai' and game.player == ai.player and game.running:
            #update the screen
            pygame.display.update()
            
            #ai methods
            row, col = ai.eval(board)
            
            game.make_move(row, col)
            
            if game.is_over():
                game.running = False
        pygame.display.update()  




