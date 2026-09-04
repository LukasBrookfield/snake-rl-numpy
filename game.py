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

        tail = self.snake.pop()

        apple_ate = False
        if self.snake[0] == self.apple_pos:
            self.snake.append(tail)
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

    def get_state_features(self) -> tuple:
        """Calculate the state vector (an numpy array) using the information provided by the game."""

        # state vector includes: 
        # 1.  danger_straight - is there danger 1 step ahead of head
        # 2.  danger_left     - is there danger 1 step left of head
        # 3.  danger_right    - is there danger 1 step right of head 
        # 4.  dir_left        - is head direction left
        # 5.  dir_right       - is head direction right
        # 6.  dir_up          - is head direction up
        # 7.  dir_down        - is head direction down
        # 8.  food_left       - is food left of head
        # 9.  food_right      - is food right of head
        # 10. food_up         - is food above head
        # 11. food_down       - is food below head

        next_pos = lambda x,y: [self.snake[0][0] + x, self.snake[0][1] + y]
        left = next_pos(-1, 0)
        right = next_pos(1, 0)
        up = next_pos(0, -1)
        down = next_pos(0, 1)

        if self.direction == Direction.LEFT:
            next_straight = left
            next_left = down
            next_right = up
        elif self.direction == Direction.RIGHT:
            next_straight = right
            next_left = up
            next_right = down
        elif self.direction == Direction.UP:
            next_straight = up
            next_left = left
            next_right = right
        else:
            next_straight = down
            next_left = right
            next_right = left

        is_danger = lambda p: (p[0] < 0) or (p[1] < 0) or (p[0] >= TILE_WIDTH) or (p[1] >= TILE_HEIGHT) or (p in self.snake)

        danger_straight = 1 if is_danger(next_straight) else 0
        danger_left = 1 if is_danger(next_left) else 0
        danger_right = 1 if is_danger(next_right) else 0

        dir_left = 1 if self.direction == Direction.LEFT else 0
        dir_right = 1 if self.direction == Direction.RIGHT else 0
        dir_up = 1 if self.direction == Direction.UP else 0
        dir_down = 1 if self.direction == Direction.DOWN else 0

        food_left = 1 if self.apple_pos[0] < self.snake[0][0] else 0
        food_right = 1 if self.apple_pos[0] > self.snake[0][0] else 0
        food_up = 1 if self.apple_pos[1] < self.snake[0][1] else 0
        food_down = 1 if self.apple_pos[1] > self.snake[0][1] else 0

        return (danger_straight,
                danger_left,
                danger_right,
                dir_left,
                dir_right,
                dir_up,
                dir_down,
                food_left,
                food_right,
                food_up,
                food_down)
