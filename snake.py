import enum
import pygame

from settings import TILE_SIZE, SNAKE_COLOUR


class Movement(enum.Enum):
    RIGHT = "RIGHT"
    LEFT = "LEFT"
    UP = "UP"
    DOWN = "DOWN"


class Snake:
    def __init__(self) -> None:
        """Initializes snake attributes."""
        self.sprite = [[TILE_SIZE[0] * 3, TILE_SIZE[1] * 5], [TILE_SIZE[0] * 2, TILE_SIZE[1] * 5]]
        self.direction = Movement.RIGHT
        self.last_movement = Movement.RIGHT
        self.delay = 30

    def update(self, apple_pos: list[int]) -> bool:
        """Updates the snake position and sprite."""
        self.get_input()

        if self.delay > 0:
            self.delay -= 1
            return False

        head_x, head_y = self.sprite[0][0], self.sprite[0][1]

        if self.direction == Movement.RIGHT:
            self.sprite.insert(0, [head_x + TILE_SIZE[0], head_y])
            self.last_movement = Movement.RIGHT
        elif self.direction == Movement.LEFT:
            self.sprite.insert(0, [head_x - TILE_SIZE[0], head_y])
            self.last_movement = Movement.LEFT
        elif self.direction == Movement.DOWN:
            self.sprite.insert(0, [head_x, head_y + TILE_SIZE[1]])
            self.last_movement = Movement.DOWN
        elif self.direction == Movement.UP:
            self.sprite.insert(0, [head_x, head_y - TILE_SIZE[1]])
            self.last_movement = Movement.UP

        self.sprite.pop()
        self.delay = 9

        if self.sprite[0] == apple_pos:
            tail = self.sprite[-1]
            if self.direction == Movement.RIGHT:
                self.sprite.append([tail[0] - TILE_SIZE[0], tail[1]])
            elif self.direction == Movement.LEFT:
                self.sprite.append([tail[0] + TILE_SIZE[0], tail[1]])
            elif self.direction == Movement.DOWN:
                self.sprite.append([tail[0], tail[1] - TILE_SIZE[1]])
            elif self.direction == Movement.UP:
                self.sprite.append([tail[0], tail[1] + TILE_SIZE[1]])
            return True
        else:
            return False

    def draw(self, win: pygame.surface.Surface) -> None:
        """Draws the snake sprite to the screen."""
        image = pygame.Surface(TILE_SIZE)
        image.fill(SNAKE_COLOUR)
        for pos in self.sprite[1:]:
            win.blit(image, pos)
        image.fill("white")
        win.blit(image, self.sprite[0])


    def get_input(self) -> None:
        """Gets the input from the player in order to update the snake."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT] and self.last_movement != Movement.LEFT:
            self.direction = Movement.RIGHT
        elif keys[pygame.K_LEFT] and self.last_movement != Movement.RIGHT:
            self.direction = Movement.LEFT
        elif keys[pygame.K_DOWN] and self.last_movement != Movement.UP:
            self.direction = Movement.DOWN
        elif keys[pygame.K_UP] and self.last_movement != Movement.DOWN:
            self.direction = Movement.UP


    def reset_to_start(self) -> None:
        """Resets the snake attributes to the starting ones so the game can be played again."""
        self.sprite = [[TILE_SIZE[0] * 3, TILE_SIZE[1] * 5], [TILE_SIZE[0] * 2, TILE_SIZE[1] * 5]]
        self.direction = Movement.RIGHT
        self.last_movement = Movement.RIGHT
        self.delay = 30
