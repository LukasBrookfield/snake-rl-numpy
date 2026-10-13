# Reinforcement learning example with Snake
- I built Snake with Python and Pygame, and trained a reinforcement learning agent to play it with a trained policy.
- Some RL techniques I applied:
- Q-learning with small state space to allow for hash table lookup of Q values
- Deep Q-Network
    - Epsilon-greedy exploration, with epsilon decay
    - My own neural network trained with stochastic gradient descent (SGD)
    - Experience replay with fixed size buffer of past steps
## Results table

| Agent | Mean apples/game | Std | Median | Max | Games |
|---|---|---|---|---|---|
| DQN | 31.91 | 10.93 | 32 | 62 | 3,000 |
| Tabular Q-learning | 23.65 | 9.15 | 23 | 60 | 3,000 |
| Random policy | 0.12 | 0.35 | 0 | 3 | 3,000 |
## How to run
```bash
# play snake for yourself:
python main.py
# train and test q-learning model
python main.py qlearning
# load and test dqn model
# if you want to train it for yourself, just delete dqn.pickle
python main.py dqn
```
