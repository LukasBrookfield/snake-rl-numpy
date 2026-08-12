import random

from tqdm import tqdm
import numpy as np

from game import SnakeGame
from gui import SnakeGameGUI
from settings import *


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

        if self._size < self.capacity:
            self._size += 1

    def sample(self, sample_size: int) -> tuple:
        """Sample n random elements from the buffer and return them as a tuple."""
        # get indices of random sample of size 'sample_size'
        sample = np.random.choice(a=self._size, size=sample_size, replace=False)

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
        self.W1 = np.random.randn(765, 256) * np.sqrt(2.0, 765)
        self.b1 = np.random.zeros(256)

        # weights and biases for 2nd layer
        self.W2 = np.random.randn(256, 128) * np.sqrt(2.0, 256)
        self.b2 = np.zeros(128)

        # weights and biases for 3rd layer
        self.W3 = np.random.randn(128, 3) * np.sqrt(2.0 / 128)
        self.b3 = np.zeros(3)

    def back_prop(self, X: np.ndarray, y: np.ndarray) -> None:
        """Performs backward propagation."""
        batch_size = X.shape[0]
        
        Z1, A1, Z2, A2, Z3 = self.forward_prop(X)
        
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
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        
    def ReLU(self, Z: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, Z)

    def ReLU_deriv(self, Z: np.ndarray) -> np.ndarray:
        """Derivative of ReLU for back propagation."""
        return (Z > 0).astype(float)
    
    def forward_prop(self, X: np.ndarray) -> tuple:
        """Performs forward propagation on neural network. X must be of shape (765,)"""
        # input layer
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.ReLU(Z1)

        # second layer
        Z2 = self.W2.dot(A1) + self.b2
        A2 = self.ReLU(Z2)

        # output layer
        Z3 = self.W3.dot(A2) + self.b3

        return (Z1, A1, Z2, A2, Z3)

    def copy_weights(self, other_network: "DQNNeuralNetwork") -> None:
        """Copy weights from other network to current network."""
        self.W1 = other_network.W1.copy()
        self.b1 = other_network.b1.copy()
        
        self.W2 = other_network.W2.copy()
        self.b2 = other_network.b2.copy()
        
        self.W3 = other_network.W3.copy()
        self.b3 = other_network.b3.copy()


class SnakeDQN:
    def __init__(self,
                 num_episodes: int = 10_000,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 50_000, 
                 sample_size: int = 64,
                 C: int = 1500,
                 step_reward: int = 0,
                 apple_reward: int = 10,
                 end_reward: int = -10):
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.buffer_capacity = buffer_capacity
        self.sample_size = sample_size
        self.C = C  # number of time steps before resetting Q_hat
        self.step_reward = step_reward
        self.apple_reward = apple_reward
        self.end_reward = end_reward

    def train(self, game: SnakeGame) -> None:
        epsilon = self.epsilon
        epsilon_min = self.epsilon_min
        epsilon_decay = self.epsilon_decay

        # initialise replay memory D
        buffer = ReplayMemoryBuffer(self.buffer_capacity)

        # initialise current C (determines steps until reset)
        C_curr = 0

        # initialise action-value function with random weights
        Q = DQNNeuralNetwork()

        # initialise target action-value function Q_hat with weights
        Q_hat = DQNNeuralNetwork()

        for _ in tqdm(range(self.num_episodes), desc="Training..."):

            # **************************** For t = 1, T do: ****************************
            
            s = self.calculate_state(game)

            # **********SAMPLING***********
            # With probability ε select a random action a
            if random.random() < epsilon:
                a = np.random.randint(0, 3)
            # otherwise select a = argmax Q(s,a)
            else:
                a = np.argmax(Q.forward_prop(s))

            # Execute action a in emulator and observe reward r_t and next state s'
            old_score = game.score
            game.next_state(a)

            # calculate immediate reward r
            r = self.step_reward
            r += game.score - old_score
            d = game.is_game_over
            r += self.end_reward if d else 0

            # Store transition (s,a,r,s',d) in D
            s_dash = self.calculate_state(game)
            buffer.add(s, a, r, s_dash, d)

            # *********TRAINING*********
            # Wait until there are enough transitions to fill sample
            if buffer.size < self.sample_size:
                continue

            # Sample random minibatch of transitions (s,a,r,s',d) from D
            s_sample, a_sample, r_sample, ns_sample, d_sample = buffer.sample(self.sample_size)
                        

            # X (input batch): (64, 765)
            # Q (output batch): (64, 3)
            ...

            # Set y = r if episode terminates
            # else r + γ max( Q_hat(S_t+1,a) )
            ...

            # Perform a gradient descent step on (y - Q(s,a))^2
            # X (input batch): (64, 765)
            ...

            # Every C steps reset Q_hat = Q
            ...

            # Decay epsilon
            

    def test(self) -> None:
        pass

    def calculate_state(self) -> np.ndarray:
        """
        Calculate state array based of game information.
        It consists of a flattened array of 3 15x17 grids:
        1. one hot encoded position of head
        2. one hot encoded positions of body
        3. one hot encoded position of apple
        """
        pass
