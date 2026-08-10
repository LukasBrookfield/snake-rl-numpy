import os
import sys
import pygame

from board import Board
from snake import Snake
from settings import WINDOW_WIDTH, WINDOW_HEIGHT, FPS


def create_highscore_file() -> None:
    if not os.path.exists("highscore.txt"):
        with open("highscore.txt", "w") as file:
            file.write("0")


def main() -> None:
    """Entry point to the program."""
    pygame.init()
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    create_highscore_file()
    snake = Snake()
    board = Board(snake.sprite)
    game_over = False

    while True:
        
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                board.save_high_score()
                pygame.quit()
                sys.exit()
        
        if not game_over:
            board.draw(win)
            apple_ate = snake.update(board.apple_pos)
            snake.draw(win)
            board.update(win, apple_ate, snake.sprite)
            game_over = board.game_over(snake.sprite)
        else:
            snake.reset_to_start()
            board.update_game_over_screen(win)
            if board.play_again:
                board.play_again = False
                game_over = False
        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
