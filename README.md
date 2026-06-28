Pirate Intelligent Agent
A deep Q-learning agent that learns to navigate an 8x8 maze and find treasure before a human player, built as part of SNHU's CS 370: Current/Emerging Trends in Computer Science.
Overview
This project simulates a treasure hunt game where a pirate NPC must learn, through trial and error, the optimal path through a maze to reach a hidden treasure. The agent has no prior knowledge of the maze layout and instead learns a policy purely through repeated interaction with the environment, using a deep Q-network (DQN) to estimate the value of each possible move from any given position.

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
What I Did vs. What Was Provided
The starting code I was given included TreasureMaze.py (the maze environment, including the maze matrix, valid action checking, reward logic, and state tracking) and GameExperience.py (the experience replay buffer, which stores past episodes and samples batches for training). The build_model() function and the overall notebook structure, including the maze visualization and the play_game() and completion_check() functions, were also provided.

What I built myself was the Q-Training Algorithm, the core training loop inside the qtrain() function. That included implementing the epsilon-greedy exploration/exploitation logic, resetting the environment at a random starting cell each epoch, looping through the game until a win, loss, or timeout, storing each step's experience in the replay buffer, calling the model's training step on sampled batches, tracking win history and win rate, and decaying epsilon over time until the agent reliably reached a 100% win rate.
Reflection: Connecting This Course to Computer Science
What do computer scientists do, and why does it matter?

Computer scientists build systems that solve real problems by breaking them down into smaller, well-defined pieces and figuring out how to represent those pieces in a way a machine can actually work with. With this project specifically, the "problem" was something a human could already do (find a path through a maze), but the work was making a machine learn to do it without being told the answer directly. That matters because most real-world problems do not come with a clean, predefined solution. Being able to design a system that learns and adapts, rather than one that just follows a fixed set of rules, is what lets computer science scale to messy, unpredictable real-world situations like logistics, robotics, or personalized recommendations.

How do I approach a problem as a computer scientist?

This project really reinforced breaking a problem down before writing any code. Before touching the Q-training loop, I had to understand the environment (the maze and its reward structure), the existing components I was given (the experience replay class and the model), and the actual goal (a 100% win rate from every starting cell, not just one). Only after that did the actual algorithm design happen: deciding how exploration and exploitation should be balanced, how experience should be stored and replayed, and how to know when the agent was actually done learning versus just lucky. I also learned to lean on debugging methodically rather than guessing. When errors came up, like the GPU/CPU conflicts or shape mismatches, the move was always to read the traceback carefully and trace it back to its root cause rather than randomly changing things, the same approach used throughout the semester.

What are my ethical responsibilities to the end user and the organization?

Even in a simple game like this, the choices I made about the reward structure and training process directly shaped how the agent behaved. Scaled up to real systems, that same idea means a computer scientist's design choices have direct consequences for the people who use the system and the organization that deploys it. My responsibility to the end user is to build systems that are transparent about their limitations rather than appearing more reliable than they are, and to actively consider whether a system's training data or reward structure could create unfair or biased outcomes for some users over others. My responsibility to the organization is to build systems that are actually maintainable and well-documented, like this README, rather than something that only makes sense to the person who wrote it. Across this whole course, the recurring theme was that capability without accountability is where AI systems get dangerous, and that responsibility does not stop once the code runs without errors.
Author
Kelly Reinersman — CS 370, Southern New Hampshire University
References
Beysolow, T. (2019). Applied reinforcement learning with Python: With OpenAI Gym, TensorFlow, and Keras. Apress.
Lamba, A. (2018). A brief introduction to reinforcement learning. freeCodeCamp.
Serengeti. (2019). Using Q-learning for pathfinding.

