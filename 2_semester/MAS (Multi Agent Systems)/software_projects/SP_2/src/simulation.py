import threading
import time

import pygame

from src import keys
from src.core.blackboard import Blackboard
from src.core.communication_agent import CommunicationAgent
from src.core.state import ClassicGameState
from src.classic.environment import ClassicTicTacToeEnvironment
from src.game.tictactoe_game import TicTacToeGame


class Simulation:
    DISPLAY_INTERVAL = 2

    def __init__(self, agents: list[CommunicationAgent], environment: ClassicTicTacToeEnvironment,
        game: TicTacToeGame,
        blackboard: Blackboard,
    ) -> None:
        self.agents = agents
        self.environment = environment
        self.game = game
        self.blackboard = blackboard
        self._threads: list[threading.Thread] = []
        self._last_grid: list[list[int]] | None = None

    def start_agent_threads(self) -> None:
        for agent in self.agents:
            thread = threading.Thread(target=agent.run, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _drain_messages(self) -> None:
        while True:
            message = self.blackboard.pop_message()
            if message is None:
                break
            print(message)

    def run(self) -> None:
        self.start_agent_threads()
        running = True

        while running:
            self._drain_messages()

            grid = self.blackboard.read(keys.BOARD)
            if grid is not None and grid != self._last_grid:
                game_over = self.blackboard.read(keys.GAME_OVER)
                winner = self.blackboard.read(keys.WINNER)
                current_turn = self.blackboard.read(keys.CURRENT_TURN)

                state = ClassicGameState(
                    grid=grid,
                    current_turn=current_turn or 1,
                    game_over=bool(game_over),
                    winner=winner or 0,
                )
                print(state.display())
                print("-------------\n")

                self.game.sync_from_grid(grid)
                self._last_grid = grid

                if game_over:
                    if winner == 0:
                        print("Game over: draw.")
                    else:
                        print(f"Game over: player {winner} wins.")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.update()
            time.sleep(self.DISPLAY_INTERVAL)

        pygame.quit()
