import pygame

from src.constants import HEIGHT, WIDTH
from src.core.blackboard import Blackboard
from src.classic.environment import ClassicTicTacToeEnvironment
from src.agents.game_master import GameMasterAgent
from src.agents.dummy_agent import DummyStrategicAgent
from src.agents.minimax_agent import MinimaxStrategicAgent
from src.simulation import Simulation
from src.game.tictactoe_game import TicTacToeGame


def run_simulation() -> None:
    blackboard = Blackboard()
    environment = ClassicTicTacToeEnvironment()

    game_master = GameMasterAgent(blackboard, environment)
    # player_one = DummyStrategicAgent(blackboard, player_id=1, environment=environment)
    player_one = MinimaxStrategicAgent(
        blackboard, player_id=1, environment=environment,
    )
    
    player_two = MinimaxStrategicAgent(
        blackboard, player_id=2, environment=environment
    )

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MAS Tic Tac Toe")
    game = TicTacToeGame(screen)

    simulation = Simulation(
        agents=[game_master, player_one, player_two],
        environment=environment,
        game=game,
        blackboard=blackboard,
    )
    simulation.run()


if __name__ == "__main__":
    run_simulation()
