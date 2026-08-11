import os
import sys
import random

import pygame as pg

from game import Direction
from settings import *


class SnakeGameGUI:
    def __init__(self, initial_move_delay: int = DEFAULT_INITIAL_MOVE_DELAY, move_delay: int = DEFAULT_MOVE_DELAY):
        """Initialize game attributes"""
        pg.init()
        self.create_highscore_file()
        self.win = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pg.time.Clock()
        self.is_game_over = False

        # SNAKE ATTRS
        self.snake_sprite = [[TILE_SIZE[0] * START_X, TILE_SIZE[1] * START_Y], [TILE_SIZE[0] * (START_X - 1), TILE_SIZE[1] * START_Y]]
        self.direction = Direction.RIGHT
        self.last_movement = Direction.RIGHT
        self.INITIAL_MOVE_DELAY = initial_move_delay
        self.MOVE_DELAY = move_delay
        self.delay = initial_move_delay

        # BOARD ATTRS
        self.apple_pos = self.generate_apple(self.snake_sprite)
        self.font = pg.font.SysFont(FONT, 70)
        self.high_score = self.get_high_score()
        self.score = 0
        self.play_again = False

    def start(self):
        """Start game loop"""
        while True:
            events = pg.event.get()
            for event in events:
                if event.type == pg.QUIT:
                    self.save_high_score()
                    pg.quit()
                    sys.exit()
            
            if not self.is_game_over:
                self.draw_board(self.win)
                apple_ate = self.update_snake(self.apple_pos)
                self.draw_snake(self.win)
                self.update_board(self.win, apple_ate, self.snake_sprite)
                self.is_game_over = self.game_over(self.snake_sprite)
            else:
                self.reset_snake_to_start()
                self.update_game_over_screen(self.win)
                if self.play_again:
                    self.play_again = False
                    self.is_game_over = False

            pg.display.update()
            self.clock.tick(FPS)

    def update_snake(self, apple_pos: list[int]) -> bool:
        """Updates the snake position and sprite."""
        self.get_input()

        if self.delay > 0:
            self.delay -= 1
            return False

        head_x, head_y = self.snake_sprite[0][0], self.snake_sprite[0][1]

        if self.direction == Direction.RIGHT:
            self.snake_sprite.insert(0, [head_x + TILE_SIZE[0], head_y])
            self.last_movement = Direction.RIGHT
        elif self.direction == Direction.LEFT:
            self.snake_sprite.insert(0, [head_x - TILE_SIZE[0], head_y])
            self.last_movement = Direction.LEFT
        elif self.direction == Direction.DOWN:
            self.snake_sprite.insert(0, [head_x, head_y + TILE_SIZE[1]])
            self.last_movement = Direction.DOWN
        elif self.direction == Direction.UP:
            self.snake_sprite.insert(0, [head_x, head_y - TILE_SIZE[1]])
            self.last_movement = Direction.UP

        self.snake_sprite.pop()
        self.delay = self.MOVE_DELAY

        if self.snake_sprite[0] == apple_pos:
            tail = self.snake_sprite[-1]
            if self.direction == Direction.RIGHT:
                self.snake_sprite.append([tail[0] - TILE_SIZE[0], tail[1]])
            elif self.direction == Direction.LEFT:
                self.snake_sprite.append([tail[0] + TILE_SIZE[0], tail[1]])
            elif self.direction == Direction.DOWN:
                self.snake_sprite.append([tail[0], tail[1] - TILE_SIZE[1]])
            elif self.direction == Direction.UP:
                self.snake_sprite.append([tail[0], tail[1] + TILE_SIZE[1]])
            return True
        else:
            return False

    def draw_snake(self, win: pg.surface.Surface) -> None:
        """Draws the snake sprite to the screen."""
        image = pg.Surface(TILE_SIZE)
        image.fill(SNAKE_COLOUR)

        for pos in self.snake_sprite[1:]:
            win.blit(image, pos)

        image.fill(SNAKE_HEAD_COLOUR)
        win.blit(image, self.snake_sprite[0])


    def get_input(self) -> None:
        """Gets the input from the player in order to update the snake."""
        keys = pg.key.get_pressed()
        if keys[pg.K_RIGHT] and self.last_movement != Direction.LEFT:
            self.direction = Direction.RIGHT
        elif keys[pg.K_LEFT] and self.last_movement != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif keys[pg.K_DOWN] and self.last_movement != Direction.UP:
            self.direction = Direction.DOWN
        elif keys[pg.K_UP] and self.last_movement != Direction.DOWN:
            self.direction = Direction.UP

    def reset_snake_to_start(self) -> None:
        """Resets the snake attributes to the starting ones so the game can be played again."""
        self.snake_sprite = [[TILE_SIZE[0] * 3, TILE_SIZE[1] * 5], [TILE_SIZE[0] * 2, TILE_SIZE[1] * 5]]
        self.direction = Direction.RIGHT
        self.last_movement = Direction.RIGHT
        self.delay = self.INITIAL_MOVE_DELAY

    def update_board(self, win: pg.surface.Surface, apple_ate: bool, snake_sprite: list[list[int]]) -> None:
        """Updates the board state."""     
        if apple_ate:
            self.score += 1
            self.apple_pos = self.generate_apple(snake_sprite)
        
        score_text = self.font.render(SCORE_TEXT + str(self.score), True, TEXT_COLOUR)
        score_text_rect = score_text.get_rect(topleft=(20, 20))
        win.blit(score_text, score_text_rect)
    
    
    def draw_board(self, win: pg.surface.Surface) -> None:
        """Draws the board on to the screen."""
        colour_1 = COLOUR_1
        colour_2 = COLOUR_2

        for y in range(0, WINDOW_HEIGHT // TILE_SIZE[1]):
            for x in range(0, WINDOW_WIDTH // TILE_SIZE[0]):
                x_pos, y_pos = x * TILE_SIZE[0], y * TILE_SIZE[1]
                surf = pg.Surface(TILE_SIZE)
                rect = surf.get_rect(topleft=(x_pos, y_pos))
                if [x_pos, y_pos] == self.apple_pos:
                    surf.fill(APPLE_COLOUR)
                else:
                    surf.fill(colour_1)
                colour_1, colour_2 = colour_2, colour_1
                win.blit(surf, rect)
        
    def generate_apple(self, snake_sprite: list[list[int]]) -> list[int]:
        """Generates a new position for the apple."""
        pos = [random.randint(0, TILE_WIDTH - 1) * TILE_SIZE[0], random.randint(0, TILE_HEIGHT - 1) * TILE_SIZE[1]]
        while pos in snake_sprite:
            pos = [random.randint(0, TILE_WIDTH - 1) * TILE_SIZE[0], random.randint(0, TILE_HEIGHT - 1) * TILE_SIZE[1]]
        return pos

    def game_over(self, snake_sprite: list[list[int]]) -> bool:
        """Calculates if the snake has lost. Returns a boolean (True if lost, False if not).""" 
        if snake_sprite[0] in snake_sprite[1:] or snake_sprite[0][0] > (WINDOW_WIDTH - TILE_SIZE[0]) or snake_sprite[0][0] < 0 or snake_sprite[0][1] > (WINDOW_HEIGHT - TILE_SIZE[1]) or snake_sprite[0][1] < 0 or len(snake_sprite) >= ((WINDOW_WIDTH / TILE_SIZE[0]) * (WINDOW_HEIGHT / TILE_SIZE[1])):
            self.apple_pos = self.generate_apple(snake_sprite)
            if self.score > self.high_score:
                self.high_score = self.score
            return True
        else:
            return False

    def update_game_over_screen(self, win: pg.surface.Surface) -> None:
        """Displays the game-over screen"""
        self.draw_board(win)
        score = self.font.render(SCORE_TEXT + str(self.score), True, TEXT_COLOUR)
        high_score = self.font.render(HIGH_SCORE_TEXT + str(self.high_score), True, TEXT_COLOUR)
        play_again = self.font.render(PLAY_AGAIN_TEXT, True, TEXT_COLOUR)
        
        score_rect = score.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        high_score_rect = high_score.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        play_again_rect = play_again.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 100))

        win.blit(score, score_rect)
        win.blit(high_score, high_score_rect)
        win.blit(play_again, play_again_rect)

        keys = pg.key.get_pressed()
        if keys[pg.K_RETURN]:
            self.save_high_score()
            self.score = 0
            self.play_again = True

    @staticmethod
    def create_highscore_file() -> None:
        if not os.path.exists(HIGH_SCORE_FILE_NAME):
            with open(HIGH_SCORE_FILE_NAME, "w") as file:
                file.write("0")

    @staticmethod
    def get_high_score() -> int:
        """Returns the high score from the 'highscore.txt' file."""
        with open(HIGH_SCORE_FILE_NAME, "r") as file:
            return int(file.read())
    
    def save_high_score(self) -> None:
        """Saves the score to the high score file if it is a new high score."""
        if self.score > self.get_high_score():
            with open(HIGH_SCORE_FILE_NAME, "w") as file:
                file.write(str(self.score))

