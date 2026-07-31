import pygame
from src.constants import *


class TicTacToeRenderer:
    def __init__(self, screen):
        self.screen = screen

    def draw_lines(self):
        self.screen.fill(BG_COLOR) #for resetting the screen
        
        pygame.draw.line(self.screen, LINE_COLOR, (SQSIZE, 0), (SQSIZE, HEIGHT), LINE_WIDTH)
        pygame.draw.line(self.screen, LINE_COLOR, (2 * SQSIZE, 0), (2 * SQSIZE, HEIGHT), LINE_WIDTH)
        pygame.draw.line(self.screen, LINE_COLOR, (0, SQSIZE), (WIDTH, SQSIZE), LINE_WIDTH)
        pygame.draw.line(self.screen, LINE_COLOR, (0, 2 * SQSIZE), (WIDTH, 2 * SQSIZE), LINE_WIDTH)

    def draw_figure(self, row, col, player):
        if player == 1:
            start_descending = (col * SQSIZE + OFFSET, row * SQSIZE + OFFSET)
            end_descending = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            pygame.draw.line(self.screen, CROSS_COLOR, start_descending, end_descending, CROSS_WIDTH)
            start_ascending = (col * SQSIZE + OFFSET, row * SQSIZE + SQSIZE - OFFSET)
            end_ascending = (col * SQSIZE + SQSIZE - OFFSET, row * SQSIZE + OFFSET)
            pygame.draw.line(self.screen, CROSS_COLOR, start_ascending, end_ascending, CROSS_WIDTH)
        
        elif player == 2:
            center = (col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2)
            pygame.draw.circle(self.screen, CIRCLE_COLOR, center, CIRCLE_RADIUS, CIRCLE_WIDTH)

    def draw_winner_line(self, player, won_type, index):
        color = CROSS_COLOR if player == 1 else CIRCLE_COLOR

        if won_type == 'vertical':
            start = (index * SQSIZE + SQSIZE // 2, 20)
            end = (index * SQSIZE + SQSIZE // 2, HEIGHT - 20)
        elif won_type == 'horizontal':
            start = (20, index * SQSIZE + SQSIZE // 2)
            end = (WIDTH - 20, index * SQSIZE + SQSIZE // 2)
        elif won_type == 'main_diag':
            start = (20, 20)
            end = (WIDTH - 20, HEIGHT - 20)
        elif won_type == 'anti_diag':
            start = (WIDTH - 20, 20)
            end = (20, HEIGHT - 20)

        pygame.draw.line(self.screen, color, start, end, LINE_WIDTH)
