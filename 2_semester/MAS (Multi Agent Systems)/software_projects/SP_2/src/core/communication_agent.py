from src.core.agent import Agent
from src.core.blackboard import Blackboard


class CommunicationAgent(Agent):
    def __init__(self, blackboard: Blackboard, name: str) -> None:
        self.blackboard = blackboard
        self.name = name

    def read(self, key: str):
        return self.blackboard.read(key)

    def write(self, key: str, value) -> None:
        self.blackboard.write(key, value, self.name)
