# Reinforcement learning example with Snake
- I built Snake with Python and Pygame, and trained a reinforcement learning agent to play it with a trained policy.
- Some RL techniques I applied:
- Q-learning with small state space to allow for hash table lookup of Q values
- Deep Q-Network
    - Epsilon-greedy exploration, with epsilon decay
    - My own neural network trained with stochastic gradient descent (SGD)
    - Experience replay with fixed size buffer of past steps
```bash
# play snake for yourself:
python main.py
# train and test q-learning model
python main.py qlearning
# load and test dqn model
# if you want to train it for yourself, just delete dqn.pickle
python main.py dqn
```
