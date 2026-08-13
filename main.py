import sys

from game import SnakeGame
from gui import SnakeGameGUI
from qlearning import SnakeQLearning
from dqn import SnakeDQN


def main() -> None:
    """Entry point to the program."""
    arg = sys.argv[1]
    if arg == "qlearning":
        ai = SnakeQLearning()
        train_game = SnakeGame()
        ai.train(train_game)

        test_game = SnakeGame()
        gui = SnakeGameGUI()
        ai.test(test_game, gui)
    elif arg == "dqn":
        ai = SnakeDQN()
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
