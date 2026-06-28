CS-370-12753-M01 Current/Emerging Trends in CS 2026 C-3 (May - Jun)
Pirate Intelligent Agent
A deep Q-learning agent that learns to navigate an 8x8 maze and find treasure before a human player, built as part of SNHU's CS 370: Current/Emerging Trends in Computer Science.
Overview
This project simulates a treasure-hunt game in which a pirate NPC must learn, through trial and error, the optimal path through a maze to reach a hidden treasure. The agent has no prior knowledge of the maze layout and instead learns a policy purely through repeated interaction with the environment, using a deep Q-network (DQN) to estimate the value of each possible move from any given position.

This is a classic pathfinding problem solved with reinforcement learning rather than traditional search algorithms like A* or Dijkstra's.
How It Works
The pirate starts at the top-left corner of the maze and can move in four directions: left, right, up, and down. At each step, the agent has to decide between:

Exploration — trying a random action to discover new information about the environment
Exploitation — choosing the action its neural network currently predicts is best

This tradeoff is controlled by an epsilon value that starts high (favoring exploration) and decays over time as the agent gains experience, eventually favoring exploitation once the model is confident in its learned policy.
Reward Structure
Outcome
Reward
Reaching the treasure
+1
Moving to an occupied/blocked cell
-0.75
Attempting to move outside the maze boundary
-0.8
Moving to a valid adjacent cell
-0.04


The small per-move penalty discourages the agent from wandering aimlessly, while the larger penalties strongly discourage invalid moves.
Algorithm
The agent uses deep Q-learning with:

A neural network that approximates Q-values (the expected future reward) for each action given the current state
Experience replay — storing past transitions in memory and training on randomly sampled batches rather than learning from each step immediately, which stabilizes training
An epsilon-greedy policy with decay for balancing exploration and exploitation
Project Structure
pirate-intelligent-agent/

├── TreasureHuntGame.ipynb     # Main notebook — environment setup, model, training loop

├── TreasureMaze.py            # Maze environment class (provided, not modified)

├── GameExperience.py          # Experience replay buffer class (provided, not modified)

└── requirements.txt           # Python dependencies
Requirements
Python 3.11+
TensorFlow / Keras
NumPy
Matplotlib (for maze visualization)

Install dependencies:

pip install -r requirements.txt

Note: If running outside a GPU-enabled environment, add the following before importing TensorFlow to avoid GPU initialization errors:

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
Running the Project
Open TreasureHuntGame.ipynb in Jupyter Notebook or JupyterLab
Run all cells in order — the notebook will:
Build and visualize the maze
Construct the neural network model
Train the agent using deep Q-learning until it reaches a 100% win rate
Run a completion_check() to confirm the agent can win from every valid starting cell
Play a full demo game and visualize the pirate's path to the treasure

Training can take anywhere from a few minutes to over 10 minutes depending on hyperparameters and hardware, since the model trains over hundreds of epochs.
Results
The trained agent consistently achieves a 100% win rate, successfully finding the treasure from every valid starting position in the maze. Training typically converges between epoch 150–550 depending on random initialization, with win rate climbing steadily as epsilon decays and the network's Q-value estimates improve.
Key Concepts Demonstrated
Reinforcement learning (Q-learning, deep Q-networks)
Neural network design with Keras/TensorFlow
Experience replay for stable training
Exploration vs. exploitation tradeoff
Pathfinding as a reinforcement learning problem
Author
Kelly Reinersman — CS 370, Southern New Hampshire University
References
Beysolow, T. (2019). Applied reinforcement learning with Python: With OpenAI Gym, TensorFlow, and Keras. Apress.
Lamba, A. (2018). A brief introduction to reinforcement learning. freeCodeCamp.
Serengeti. (2019). Using Q-learning for pathfinding.

