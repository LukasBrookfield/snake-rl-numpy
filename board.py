import random
import pygame

from settings import WINDOW_WIDTH, WINDOW_HEIGHT, TILE_SIZE, APPLE_COLOUR, TEXT_COLOUR


class Board:
    def __init__(self, snake_sprite: list[list[int]]) -> None:
        """Initializes board attributes."""
        self.apple_pos = self.generate_apple(snake_sprite)
        self.font = pygame.font.SysFont("Sans Serif", 70)
        self.high_score = self.get_high_score()
        self.score = 0
        self.play_again = False

    def update(self, win: pygame.surface.Surface, apple_ate: bool, snake_sprite: list[list[int]]) -> None:
        """Updates the board state."""     
        if apple_ate:
            self.score += 1
            self.apple_pos = self.generate_apple(snake_sprite)
        
        score_text = self.font.render("Score: " + str(self.score), True, TEXT_COLOUR)
        score_text_rect = score_text.get_rect(topleft=(20, 20))
        win.blit(score_text, score_text_rect)


    def draw(self, win: pygame.surface.Surface) -> None:
        """Draws the board on to the screen."""
        current_colour = "dark green"
        other_colour = "green"

        for y in range(0, WINDOW_HEIGHT // TILE_SIZE[1]):
            for x in range(0, WINDOW_WIDTH // TILE_SIZE[0]):
                x_pos, y_pos = x * TILE_SIZE[0], y * TILE_SIZE[1]
                surf = pygame.Surface(TILE_SIZE)
                rect = surf.get_rect(topleft=(x_pos, y_pos))
                if [x_pos, y_pos] == self.apple_pos:
                    surf.fill(APPLE_COLOUR)
                else:
                    surf.fill(current_colour)
                current_colour, other_colour = other_colour, current_colour
                win.blit(surf, rect)
        
    def generate_apple(self, snake_sprite: list[list[int]]) -> list[int]:
        """Generates a new position for the apple."""
        pos = [random.randint(0, 10) * TILE_SIZE[0], random.randint(0, 10) * TILE_SIZE[1]]
        while pos in snake_sprite:
            pos = [random.randint(0, 10) * TILE_SIZE[0], random.randint(0, 10) * TILE_SIZE[1]]
        return pos

    def game_over(self, snake_sprite: list[list[int]]) -> bool:
        """Calculates if the snake has lost. Returns a boolean (True if lost, False if not)."""
        if snake_sprite[0] in snake_sprite[1:] or snake_sprite[0][0] > (WINDOW_WIDTH - TILE_SIZE[0])or snake_sprite[0][0] < 0 or snake_sprite[0][1] > (WINDOW_HEIGHT - TILE_SIZE[1]) or snake_sprite[0][1] < 0:
            self.apple_pos = self.generate_apple(snake_sprite)
            if self.score > self.high_score:
                self.high_score = self.score
            return True
        else:
            return False

    def update_game_over_screen(self, win: pygame.surface.Surface) -> None:
        """Displays the game over screen"""
        self.draw(win)
        score = self.font.render("Score: " + str(self.score), True, TEXT_COLOUR)
        high_score = self.font.render("High Score: " + str(self.high_score), True, TEXT_COLOUR)
        play_again = self.font.render("Enter ENTER to play again!", True, TEXT_COLOUR)
        
        score_rect = score.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 100))
        high_score_rect = high_score.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        play_again_rect = play_again.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 100))

        win.blit(score, score_rect)
        win.blit(high_score, high_score_rect)
        win.blit(play_again, play_again_rect)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            self.save_high_score()
            self.score = 0
            self.play_again = True

    @staticmethod
    def get_high_score() -> int:
        """Returns the high score from the 'highscore.txt' file."""
        with open("highscore.txt", "r") as file:
            return int(file.read())
    
    def save_high_score(self) -> None:
        """Saves the score to the high score file if it is a new high score."""
        if self.score > self.get_high_score():
            with open("highscore.txt", "w") as file:
                file.write(str(self.score))
