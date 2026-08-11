import sys

from game import SnakeGame
from gui import SnakeGameGUI
from rl import SnakeAI


def main() -> None:
    """Entry point to the program."""
    if "train" in sys.argv:
        ai = SnakeAI()
        train_game = SnakeGame()
        ai.train_q_learning(train_game)

        test_game = SnakeGame()
        ai.test(test_game)
    else:
        game = SnakeGameGUI()
        game.start()


if __name__ == "__main__":
    main()
