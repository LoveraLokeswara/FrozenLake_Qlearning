# FrozenLake Q-Learning Implementation

A Python implementation of the Q-learning reinforcement learning algorithm applied to OpenAI Gymnasium's FrozenLake environment. This project demonstrates tabular Q-learning with epsilon-greedy exploration strategy on a 4x4 grid world.

## Overview

The FrozenLake environment is a classic grid world problem where an agent must navigate from the starting position (S) to the goal (G) while avoiding holes (H) on a frozen lake. This implementation uses Q-learning to train an agent that learns optimal actions for each state.

## Environment Details
- **Grid Size**: 4x4
- **Environment**: Non-slippery (deterministic)
- **States**: 16 positions on the grid
- **Actions**: 4 (Left, Down, Right, Up)
- **Goal**: Reach the goal state while avoiding holes

## Setup and Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv .FLvenv
source .FLvenv/bin/activate  # On Windows: .FLvenv\Scripts\activate
```

## Usage

Run the complete training and testing pipeline:
```bash
python main.py
```

The script will:
1. Train the Q-learning agent for 15,000 episodes
2. Save the trained Q-table as both `.pkl` and `.csv` files
3. Generate a performance plot (`frozenlake_4x4.png`)
4. Test the trained agent for 10 episodes with visualization

## Project Files

- `main.py` - Main implementation of Q-learning algorithm
- `frozenlake_4x4.pkl` - Trained Q-table (binary format for loading)
- `frozenlake_4x4_qtable.csv` - Trained Q-table (human-readable CSV format)
- `frozenlake_4x4.png` - Training progress visualization
- `requirements.txt` - Project dependencies
- `.gitignore` - Git ignore rules

## Algorithm Parameters

- **Learning Rate**: 0.9 (reduced to 0.0001 after exploration ends)
- **Discount Factor**: 0.9
- **Epsilon**: 1.0 → 0.0 (linear decay with rate 0.0001)
- **Training Episodes**: 15,000
- **Max Steps per Episode**: 100

## Results

The trained agent learns to successfully navigate the FrozenLake environment, with performance metrics tracked and visualized in the generated plot showing success rate over training episodes.

---
testt