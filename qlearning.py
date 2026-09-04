import random
from collections import defaultdict

from tqdm import tqdm
import numpy as np

from game import SnakeGame
from gui import SnakeGameGUI
from settings import *


class SnakeQLearning:
    # actions are relative to the direction of the snake head direction:
    # left, right, straight
    NUM_ACTIONS = 3
    STEP_REWARD = 0
    EAT_APPLE_REWARD = 10
    DIE_REWARD = -10
    MAX_STEPS_WITHOUT_APPLE = 2 * TILE_WIDTH * TILE_HEIGHT

    def __init__(self, 
                 num_episodes: int = 10_000,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 alpha: float = 0.1) -> None:
        """Initialise snake AI attrs"""
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha

        # a dictionary where:
        # key   -> state vector
        # value -> expected return of each action at that state.
        #          index | meaning
        #            0   | straight
        #            1   | left
        #            2   | right
        #
        # as there are 2^11 = 2048 unique values for the state vector, there are 2048 keys in the dictionary
        self.q_table = defaultdict(lambda: np.zeros(self.NUM_ACTIONS))

    def train(self, game: SnakeGame) -> None:
        """Trains the snake AI using Q-learning"""
        epsilon = self.epsilon
        epsilon_min = self.epsilon_min     
        epsilon_decay = self.epsilon_decay

        for _ in tqdm(range(self.num_episodes), desc="Training..."):
            game.reset_snake_to_start()
            game.is_game_over = False
            steps_since_apple = 0

            while not game.is_game_over:
                # calculate current state vector using game instance
                s = self.calculate_state_vector(game)

                # get Q(s,a) values for current state
                q_vals = self.q_table[s]

                # pick next action using ε-greedy with current policy
                if random.random() < epsilon:
                    # pick random action
                    a = random.randint(0, 2)
                else:
                    # pick next action greedily
                    a = np.argmax(q_vals)

                # let game play out
                old_score = game.score
                game.next_state(a)

                # add small -ve penalty for each step to encourage getting rewards quickly
                r = self.STEP_REWARD

                # add +10 reward for eating apple
                if game.score >= old_score + 1:
                    r += self.EAT_APPLE_REWARD
                    steps_since_apple = 0
                else:
                    steps_since_apple += 1

                # subtract -10 reward for ending game
                if game.is_game_over:
                    r += self.DIE_REWARD
                    # if game has ended, no more rewards are possible
                    next_max_q = 0.0
                else:
                    # calculate the state after action a
                    s_next = self.calculate_state_vector(game)

                    # look up the next state in the table and find its max value
                    next_max_q = np.max(self.q_table[s_next])


                # update policy using Q-learning formula
                # Q(s_t,a_t) = Q(s_t,a_t) + α(r_t+1 + max( γQ(S_t+1,a') - Q(s_t,a_t)))
                q = q_vals[a]
                q_vals[a] = q + self.alpha * (r + self.discount_factor * next_max_q - q)

                # stop episode if snake is making no progress
                if steps_since_apple >= self.MAX_STEPS_WITHOUT_APPLE:
                    break

            # decay epsilon at end of episode
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay


    def test(self, game: SnakeGame, gui: SnakeGameGUI, num_games: int = 2) -> None:
        """Visualises the agent's policy in a real game."""
        for _ in range(num_games):
            game.reset_snake_to_start()
            gui.reset_snake_to_start()
            game.is_game_over = False
            steps_since_apple = 0

            while not game.is_game_over:
                # choose optimal action according to policy
                s = self.calculate_state_vector(game)
                q_vals = self.q_table[s]
                a = np.argmax(q_vals)

                # advance game to next state
                old_score = game.score
                game.next_state(a)

                # show the current state of the game on the screen
                gui.visualise(game)

                # stop episode if snake is making no progress
                steps_since_apple = 0 if game.score > old_score else steps_since_apple + 1
                if steps_since_apple >= self.MAX_STEPS_WITHOUT_APPLE:
                    break

    @staticmethod
    def calculate_state_vector(game: SnakeGame) -> tuple:
        """Calculate the state vector (a tuple of 11 features) using the information provided by the game."""
        return game.get_state_features()
