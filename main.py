import sys

from game import SnakeGame
from gui import SnakeGameGUI
from qlearning import SnakeQLearning


def main() -> None:
    """Entry point to the program."""
    if "train" in sys.argv:
        ai = SnakeQLearning()
        train_game = SnakeGame()
        ai.train_q_learning(train_game)

        test_game = SnakeGame()
        gui = SnakeGameGUI()
        ai.test(test_game, gui)
    else:
        game = SnakeGameGUI()
        game.start()


if __name__ == "__main__":
    main()
