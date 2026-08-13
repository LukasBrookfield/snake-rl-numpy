import random
import pickle as pkl

from tqdm import tqdm
import numpy as np

from game import SnakeGame
from gui import SnakeGameGUI
from settings import *


class SnakeDQN:
    def __init__(self,
                 num_episodes: int = 150,  # 15_000
                 discount_factor: float = 0.99,  # 0.99
                 epsilon: float = 1.0,  # 1.0
                 epsilon_min: float = 0.01,  # 0.01
                 epsilon_decay: float = 0.995,  # 0.9995
                 buffer_capacity: int = 5_000,  # 50_000
                 sample_size: int = 6,  # 64
                 C: int = 15,  # 1500
                 step_reward: int = 0,
                 apple_reward: int = 10,
                 end_reward: int = -10):
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_capacity = buffer_capacity
        self.sample_size = sample_size
        self.C = C  # number of time steps before resetting Q_hat
        self.step_reward = step_reward
        self.apple_reward = apple_reward
        self.end_reward = end_reward

        self.Q = None  # action value function

    def train(self, game: SnakeGame) -> None:
        epsilon = self.epsilon
        epsilon_min = self.epsilon_min
        epsilon_decay = self.epsilon_decay

        step_count = 0

        # initialise replay memory D
        buffer = ReplayMemoryBuffer(self.buffer_capacity)

        # initialise current C (determines steps until reset)
        C_curr = 0

        # initialise action-value function with random weights
        self.Q = DQNNeuralNetwork()

        # initialise target action-value function Q_hat with weights
        Q_hat = DQNNeuralNetwork()

        for _ in tqdm(range(self.num_episodes), desc="Training..."):

            game.reset_snake_to_start()
            game.is_game_over = False

            while not game.is_game_over:
                s = self.calculate_state_vector(game)

                # **********SAMPLING***********                
                # With probability ε select a random action aself.Q.predict(s)
                if random.random() < epsilon:
                    a = np.random.randint(0, 3)
                # otherwise select a = argmax Q(s,a)
                else:
                    a = np.argmax(self.Q.predict(s))

                # Execute action a in emulator and observe reward r_t and next state s'
                old_score = game.score
                game.next_state(a)

                # calculate immediate reward r
                r = self.step_reward
                r += self.apple_reward * (game.score - old_score)
                d = game.is_game_over
                r += self.end_reward if d else 0

                # Store transition (s,a,r,s',d) in D
                s_dash = self.calculate_state_vector(game)
                buffer.add(s, a, r, s_dash, d)

                # *********TRAINING*********
                # Wait until there are enough transitions to fill sample
                step_count += 1
                if buffer.size < self.sample_size or step_count % 4 != 0:
                    continue

                # Sample random minibatch of transitions (s,a,r,s',d) from D
                states, actions, rewards, next_states, is_done = buffer.sample(self.sample_size)
                f_prop = self.Q.forward_prop(states)
                Q_pred = f_prop[4]
                Q_next = Q_hat.predict(next_states)
                y = np.copy(Q_pred)
                max_Q_next = np.max(Q_next, axis=1)
                target_values = rewards + self.discount_factor * max_Q_next * (1 - is_done.astype(int))
                batch_indices = np.arange(self.sample_size)
                y[batch_indices, actions] = target_values

                # Perform a gradient descent step
                self.Q.back_prop(states, y, f_prop)

                # Every C steps reset Q_hat = Q
                C_curr += 1            
                if C_curr == self.C:
                    Q_hat.copy_weights(self.Q)
                    C_curr = 0


            # decay epsilon at end of episode
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay


        with open("dqn.pickle", "wb") as file:
            pkl.dump(self.Q, file)


    def test(self, game: SnakeGame, gui: SnakeGameGUI, num_games: int = 3) -> None:
        """Visualises the agent's policy in a real game."""
        for _ in range(num_games):
            game.reset_snake_to_start()
            gui.reset_snake_to_start()
            game.is_game_over = False

            while not game.is_game_over:
                # choose optimal action according to policy
                s = self.calculate_state_vector(game)
                a = np.argmax(self.Q.predict(s))

                # advance game to next state
                game.next_state(a)

                # show the current state of the game on the screen
                gui.visualise(game)

    def calculate_state_vector(self, game: SnakeGame) -> np.ndarray:
        """
        Calculate state vector based of game information.
        It consists of a flattened array of 3 15x17 grids (so length 765):
        1. one hot encoded position of head
        2. one hot encoded positions of body
        3. one hot encoded position of apple
        """
        head_one_hot = np.zeros(255)
        hx, hy = game.snake[0]
        if 0 <= hx < TILE_WIDTH and 0 <= hy < TILE_HEIGHT:
            head_one_hot[hx + TILE_WIDTH * hy] = 1

        body_one_hot = np.zeros(255)
        for bx, by in game.snake[1:]:
            if 0 <= bx < TILE_WIDTH and 0 <= by < TILE_HEIGHT:
                body_one_hot[bx + TILE_WIDTH * by] = 1

        apple_one_hot = np.zeros(255)
        ax, ay = game.apple_pos
        if 0 <= ax < TILE_WIDTH and 0 <= ay < TILE_HEIGHT:
            apple_one_hot[ax + TILE_WIDTH * ay] = 1

        return np.hstack([head_one_hot, body_one_hot, apple_one_hot])


class ReplayMemoryBuffer:
    def __init__(self, capacity: int) -> None:
        """Initialise buffer with a fixed capacity."""
        self.capacity = capacity
        self.size = 0
        self._curr_pos = 0
        self._states = np.empty(shape=(capacity, 765), dtype=float)
        self._actions = np.empty(shape=capacity, dtype=int)
        self._rewards = np.empty(shape=capacity, dtype=float)
        self._next_states = np.empty(shape=(capacity, 765), dtype=float)
        self._is_done = np.empty(shape=capacity, dtype=bool)

    def add(self, s: np.ndarray, a: int, r: float, s_dash: np.ndarray, d: bool) -> None:
        """Add a transition tuple (s,a,r,s',d) to the buffer"""
        self._states[self._curr_pos] = s
        self._actions[self._curr_pos] = a
        self._rewards[self._curr_pos] = r
        self._next_states[self._curr_pos] = s_dash
        self._is_done[self._curr_pos] = d

        self._curr_pos += 1
        if self._curr_pos >= self.capacity:
            self._curr_pos = 0

        if self.size < self.capacity:
            self.size += 1

    def sample(self, sample_size: int) -> tuple:
        """Sample n random elements from the buffer and return them as a tuple."""
        # get indices of random sample of size 'sample_size'
        sample = np.random.choice(a=self.size, size=sample_size, replace=False)

        # access samples using indices
        states_sample = self._states[sample]
        actions_sample = self._actions[sample]
        rewards_sample = self._rewards[sample]
        next_states_sample = self._next_states[sample]
        is_done_sample = self._is_done[sample]

        return (states_sample, actions_sample, rewards_sample, next_states_sample, is_done_sample)


class DQNNeuralNetwork:
    def __init__(self, alpha: float = 0.001):
        # learning rate
        self.alpha = alpha

        # He initialisation for weights (optimal for ReLU)
        # weights and biases for 1st layer
        self.W1 = np.random.randn(765, 256) * np.sqrt(2.0 / 765)
        self.b1 = np.zeros(256)

        # weights and biases for 2nd layer
        self.W2 = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros(128)

        # weights and biases for 3rd layer
        self.W3 = np.random.randn(128, 3) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros(3)

    def predict(self, s: np.ndarray) -> np.ndarray:
        """Uses forward propagation to return Q values."""
        return self.forward_prop(s)[4]

    def back_prop(self, X: np.ndarray, y: np.ndarray, f_prop: tuple) -> None:
        """Performs backward propagation."""
        batch_size = X.shape[0]
        
        Z1, A1, Z2, A2, Z3 = f_prop
        
        # output layer gradients (derivative of MSE)
        dZ3 = (Z3 - y) / batch_size
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0)

        # hidden layer 2 gradients
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.ReLU_deriv(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0)

        # hidden layer 1 gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.ReLU_deriv(Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0)

        # update weights using gradient descent
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        
        self.W3 -= self.alpha * dW3
        self.b3 -= self.alpha * db3
        
    def ReLU(self, Z: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, Z)

    def ReLU_deriv(self, Z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU for back propagation."""
        return (Z > 0).astype(float)
    
    def forward_prop(self, X: np.ndarray) -> tuple:
        """Performs forward propagation on neural network. X must be of shape (765,)"""
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.ReLU(Z1)

        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.ReLU(Z2)
        
        Z3 = np.dot(A2, self.W3) + self.b3

        return (Z1, A1, Z2, A2, Z3)

    def copy_weights(self, other_network: "DQNNeuralNetwork") -> None:
        """Copy weights from other network to current network."""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()
        
        self.W3 = other_network.W3.copy()
        self.b3 = other_network.b3.copy()
