import os
import sys

from game import SnakeGame
from gui import SnakeGameGUI
from qlearning import SnakeQLearning
from dqn import SnakeDQN
from settings import DQN_MODEL_FILE_NAME


def main() -> None:
    """Entry point to the program."""
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "qlearning":
        ai = SnakeQLearning()
        train_game = SnakeGame()
        ai.train(train_game)

        test_game = SnakeGame()
        gui = SnakeGameGUI()
        ai.test(test_game, gui)
    elif arg == "dqn":
        ai = SnakeDQN()
        if os.path.exists(DQN_MODEL_FILE_NAME):
            ai.load()
        else:
            train_game = SnakeGame()
            ai.train(train_game)

        test_game = SnakeGame()
        gui = SnakeGameGUI()
        ai.test(test_game, gui)
    else:
        game = SnakeGameGUI()
        game.start()


if __name__ == "__main__":
    main()
