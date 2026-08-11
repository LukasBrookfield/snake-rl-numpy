import enum
import random

from settings import *


class Direction(enum.Enum):
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    UP = "UP"
    DOWN = "DOWN"


class SnakeGame:
    def __init__(self) -> None:
        """Initialize game attributes"""
        self.snake = [[START_X, START_Y], [START_X - 1, START_Y]]
        self.direction = Direction.RIGHT
        self.apple_pos = self.generate_apple()
        self.score = 0
        self.is_game_over = False

    def next_state(self, action: int) -> None:
        """Progresses to next state of game."""
        if not self.is_game_over:
            self.update(action)
            self.is_game_over = self.check_game_over()
        else:
            self.reset_snake_to_start()

    def reset_snake_to_start(self) -> None:
        """Resets the snake attributes to the starting ones so the game can be played again."""
        self.snake = [[START_X, START_Y], [START_X - 1, START_Y]]        
        self.direction = Direction.RIGHT
        self.score = 0
        self.apple_pos = self.generate_apple()

    def update(self, action: int) -> None:
        """Updates the board state. Actions: 0 -> straight, 1 -> left, 2 -> right"""
        left_action_map = {
            Direction.LEFT: Direction.DOWN,
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.DOWN: Direction.RIGHT}
        right_action_map = {
            Direction.LEFT: Direction.UP,
            Direction.RIGHT: Direction.DOWN,
            Direction.UP: Direction.RIGHT,
            Direction.DOWN: Direction.LEFT}
        if action == 1:
            self.direction = left_action_map[self.direction]
        elif action == 2:
            self.direction = right_action_map[self.direction]

        head_x, head_y = self.snake[0][0], self.snake[0][1]
        if self.direction == Direction.RIGHT:
            self.snake.insert(0, [head_x + 1, head_y])
        elif self.direction == Direction.LEFT:
            self.snake.insert(0, [head_x - 1, head_y])
        elif self.direction == Direction.DOWN:
            self.snake.insert(0, [head_x, head_y + 1])
        elif self.direction == Direction.UP:
            self.snake.insert(0, [head_x, head_y - 1])

        self.snake.pop()

        apple_ate = False
        if self.snake[0] == self.apple_pos:
            tail = self.snake[-1]
            if self.direction == Direction.RIGHT:
                self.snake.append([tail[0] - 1, tail[1]])
            elif self.direction == Direction.LEFT:
                self.snake.append([tail[0] + 1, tail[1]])
            elif self.direction == Direction.DOWN:
                self.snake.append([tail[0], tail[1] - 1])
            elif self.direction == Direction.UP:
                self.snake.append([tail[0], tail[1] + 1])
            apple_ate = True

        if apple_ate:
            self.score += 1
            self.apple_pos = self.generate_apple()
        
    def generate_apple(self) -> list[int]:
        """Generates a new position for the apple."""
        pos = [random.randint(0, TILE_WIDTH - 1), random.randint(0, TILE_HEIGHT - 1)]
        while pos in self.snake:
            pos = [random.randint(0, TILE_WIDTH - 1), random.randint(0, TILE_HEIGHT - 1)]
        return pos

    def check_game_over(self) -> bool:
        """Calculates if the snake has lost. Returns a boolean (True if lost, False if not).""" 
        head_x = self.snake[0][0]
        head_y = self.snake[0][1]

        out_of_bounds = head_x < 0 or head_x >= TILE_WIDTH or head_y < 0 or head_y >= TILE_HEIGHT
        
        hit_self = self.snake[0] in self.snake[1:]
        
        board_filled = len(self.snake) >= (TILE_WIDTH * TILE_HEIGHT)

        return out_of_bounds or hit_self or board_filled
