import random
import pickle as pkl

from tqdm import tqdm
import numpy as np

from game import SnakeGame, NUM_STATE_FEATURES
from gui import SnakeGameGUI
from settings import *


class SnakeDQN:
    def __init__(self,
                 num_episodes: int = 15_000,  # 15_000
                 discount_factor: float = 0.99,  # 0.99
                 epsilon: float = 1.0,  # 1.0
                 epsilon_min: float = 0.01,  # 0.01
                 epsilon_decay: float = 0.9995,  # 0.9995
                 buffer_capacity: int = 50_000,  # 50_000
                 sample_size: int = 64,  # 64
                 C: int = 1500,  # 1500
                 step_reward: float = 0.0,  # 0
                 apple_reward: float = 10,  # 10
                 end_reward: float = -10,  # -10
                 max_steps_without_apple: int = 2 * TILE_WIDTH * TILE_HEIGHT):  # 510
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
        # abandon an episode that goes this long without eating
        self.max_steps_without_apple = max_steps_without_apple  

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
        Q_hat.copy_weights(self.Q)

        for _ in tqdm(range(self.num_episodes), desc="Training..."):

            game.reset_snake_to_start()
            game.is_game_over = False
            steps_since_apple = 0

            while not game.is_game_over:
                s = self.calculate_state_vector(game)

                # **********SAMPLING***********                
                # With probability ε select a random action
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

                # Give up on an episode that is making no progress
                steps_since_apple = 0 if game.score > old_score else steps_since_apple + 1
                if steps_since_apple >= self.max_steps_without_apple:
                    break

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


        self.save()

    def save(self, file_name: str = DQN_MODEL_FILE_NAME) -> None:
        """Saves the trained action-value function so it can be replayed without retraining."""
        with open(file_name, "wb") as file:
            pkl.dump(self.Q, file)

    def load(self, file_name: str = DQN_MODEL_FILE_NAME) -> None:
        """Loads an action-value function saved by a previous training run."""
        with open(file_name, "rb") as file:
            self.Q = pkl.load(file)

    def test(self, game: SnakeGame, gui: SnakeGameGUI, num_games: int = 3) -> None:
        """Visualises the agent's policy in a real game."""
        if self.Q is None:
            raise RuntimeError("there is no model to test: call train() or load() first")

        for _ in range(num_games):
            game.reset_snake_to_start()
            gui.reset_snake_to_start()
            game.is_game_over = False
            steps_since_apple = 0

            while not game.is_game_over:
                # choose optimal action according to policy
                s = self.calculate_state_vector(game)
                a = np.argmax(self.Q.predict(s))

                # advance game to next state
                old_score = game.score
                game.next_state(a)

                # show the current state of the game on the screen
                gui.visualise(game)

                # stop watching a snake that is just going round in circles
                steps_since_apple = 0 if game.score > old_score else steps_since_apple + 1
                if steps_since_apple >= self.max_steps_without_apple:
                    break

    def calculate_state_vector(self, game: SnakeGame) -> np.ndarray:
        """
        Calculate state vector based of game information.
        It is the same 11 features the tabular agent uses, as an array of floats:
        3 danger flags, 4 heading flags and 4 apple direction flags.
        """
        return np.array(game.get_state_features(), dtype=float)


class ReplayMemoryBuffer:
    def __init__(self, capacity: int) -> None:
        """Initialise buffer with a fixed capacity."""
        self.capacity = capacity
        self.size = 0
        self._curr_pos = 0
        self._states = np.empty(shape=(capacity, NUM_STATE_FEATURES), dtype=float)
        self._actions = np.empty(shape=capacity, dtype=int)
        self._rewards = np.empty(shape=capacity, dtype=float)
        self._next_states = np.empty(shape=(capacity, NUM_STATE_FEATURES), dtype=float)
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
        sample = np.random.randint(0, self.size, size=sample_size)

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
        self.W1 = np.random.randn(NUM_STATE_FEATURES, 256) * np.sqrt(2.0 / NUM_STATE_FEATURES)
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
        """Performs forward propagation on neural network. X must be of shape (NUM_STATE_FEATURES,)"""
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
