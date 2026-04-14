# DQN Robot Navigation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A Deep Q-Network (DQN) implementation for training a robot to navigate grid environments, collect goals, and avoid obstacles. Developed as a Bachelor's thesis in Computer Engineering.

## Demo Videos

### Training Demo

This video demonstrates the robot's training across two phases. In the first phase, the robot leverages previously learned knowledge. In the second phase, it uses transfer learning for the first time in that environment. The training is challenging—the robot nearly fails, coming just one collision away from game over.

[Watch the training demo video](https://github.com/Massi99RM/dqn-robot-navigation/releases/tag/demo)

### Testing Demo

This video demonstrates the robot testing on the final three phases using the fully trained model. Despite never encountering the obstacle-filled phase two or phase three, the model successfully completes all three phases.

[Watch the testing demo video](https://github.com/Massi99RM/dqn-robot-navigation/releases/tag/demo2)


## Overview

This project implements a reinforcement learning agent that learns to navigate through increasingly complex grid environments. The robot must collect all goals while avoiding obstacles and staying within move limits. Training progresses through six phases with escalating difficulty, using **transfer learning** to carry knowledge from simpler to more complex environments.

### Key Results

**Training results** (full simulation runs):

| Phase | Simulations | Win Rate | Avg Moves | Avg Collisions |
|-------|-------------|----------|-----------|----------------|
| Phase 1 (5×5) | 300 | 76.0% | 11.7 | 1.6 |
| Phase 1b (5×5 + obstacles) | 100 | 93.0% | 10.6 | 1.4 |
| Phase 2 (6×6) | 50 | 98.0% | 23.6 | 0.8 |
| Phase 2b (6×6 + obstacles) | 200 | 99.5% | 20.0 | 1.2 |
| Phase 3 (7×7) | 100 | 100.0% | 35.8 | 1.3 |
| Phase 3b (7×7 + obstacles) | 400 | 72.0% | 39.5 | 5.3 |

**Fine-tuning results** (10-simulation evaluation after training):

| Phase | Win Rate | Avg Moves | Avg Collisions |
|-------|----------|-----------|----------------|
| Phase 1 (5×5) | 100% | 10.4 | 0.4 |
| Phase 1b (5×5 + obstacles) | 100% | 9.0 | 0.5 |
| Phase 2 (6×6) | 100% | 22.9 | 1.5 |
| Phase 2b (6×6 + obstacles) | 100% | 18.3 | 0.9 |
| Phase 3 (7×7) | 90% | 33.1 | 1.0 |
| Phase 3b (7×7 + obstacles) | 90% | 39.2 | 3.3 |

The model achieves **≥90% win rate across all phases during fine-tuning evaluation**. Training win rates vary, with the most challenging phase (7×7 grid + 6 obstacles) reaching 72.0% over 400 simulations. Earlier phases benefit significantly from transfer learning, with Phases 2–3 (no obstacles) achieving 98–100% during training.

## How It Works

### State Representation (15 features)

The robot perceives its environment through a 15-dimensional state vector:

| Feature | Description |
|---------|-------------|
| Position (2) | Normalized X, Y coordinates |
| Adjacent cells (4) | Obstacle/border detection in each direction |
| Collision count (1) | Normalized collision history |
| Goals remaining (1) | Ratio of remaining goals |
| Steps since goal (1) | Time since last goal collection |
| Loop detection (2) | Flags for stuck/repetitive behavior |
| Goal direction (2) | Normalized vector to nearest goal |
| Goal distance (1) | Manhattan distance to nearest goal |
| Obstacle proximity (1) | Minimum distance to nearest obstacle |

### Neural Network Architecture

```
Input (15) → FC(128) → ReLU → Dropout(0.2)
          → FC(128) → ReLU → Dropout(0.2)
          → FC(64)  → ReLU → Dropout(0.2)
          → FC(4)   → Q-values (Up, Down, Left, Right)
```

### Reward System

| Event | Reward |
|-------|--------|
| Goal collected | +50 (+ efficiency bonus up to +40) |
| All goals completed | +200 |
| Each step | -0.1 |
| Collision | -30 to -50 (escalating) |
| Loop detected | -5 |
| Repetitive pattern | -3 |
| Revisiting position | -40 |
| Exploring new areas | +0.5 per unique position |
| Defeat (collisions) | -100 |
| Defeat (out of moves) | -50 |

### Training Phases

| Phase | Grid | Internal Obstacles | Max Moves | Max Collisions |
|-------|------|-------------------|-----------|----------------|
| 1 | 5×5 | 0 | 35 | 4 |
| 1b | 5×5 | 2 | 50 | 5 |
| 2 | 6×6 | 0 | 65 | 5 |
| 2b | 6×6 | 4 | 85 | 7 |
| 3 | 7×7 | 0 | 100 | 6 |
| 3b | 7×7 | 6 | 80 | 10 |

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Python | Industry standard for ML |
| ML Framework | PyTorch | Flexible tensor operations and autograd |
| Visualization | matplotlib | Real-time grid animation |
| Math | NumPy | State vector computation |

## Tech Decisions

**Progressive Curriculum Learning:** 6 training phases with increasing grid sizes (5×5 → 6×6 → 7×7) and obstacle complexity. Each phase builds upon the previous one via transfer learning, accelerating learning on harder environments.

**Custom Reward Shaping:** A sophisticated reward system encourages efficient pathfinding and discourages repetitive behavior, with escalating collision penalties and loop detection.

**Experience Replay + Target Network:** Experience replay breaks correlation between consecutive samples. A separate target network stabilizes training by providing consistent Q-value targets.

**Empirical Hyperparameter Tuning:** The hyperparameters and simulation counts were determined through empirical experimentation rather than theoretical derivation. This iterative approach is common in RL, where the interaction between environment complexity, reward shaping, and network capacity often defies purely analytical solutions. Training was repeated across 10 complete cycles with consistent results.

## Project Structure

```
dqn-robot-navigation/
│
├── robot.py                    # Robot class with state representation and movement logic
├── grid.py                     # Grid creation and obstacle configuration
├── dqn_network.py              # Neural network architecture (4-layer MLP)
├── dqn_agent.py                # DQN agent with experience replay and target network
├── experience.py               # Replay buffer implementation
│
├── phase_one.py                # Phase 1: 5×5 grid, borders only
├── phase_one_obstacles.py      # Phase 1b: 5×5 grid + 2 internal obstacles
├── phase_two.py                # Phase 2: 6×6 grid, borders only
├── phase_two_obstacles.py      # Phase 2b: 6×6 grid + 4 internal obstacles
├── phase_three.py              # Phase 3: 7×7 grid, borders only
├── phase_three_obstacles.py    # Phase 3b: 7×7 grid + 6 internal obstacles
│
├── test_robot.py               # Testing interface for trained models
└── README.md
└── Trained models (`.pth`), replay buffers (`.pkl`), and simulation counters (`simulation_count_*.txt`) are generated during training and not versioned in the repository.
```

## How to Run

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/Massi99RM/dqn-robot-navigation.git
cd dqn-robot-navigation

# Install dependencies
pip install torch numpy matplotlib
```

### Training

Training must be done sequentially, as each phase uses transfer learning from the previous one:

```bash
# Start with Phase 1
python phase_one.py

# After sufficient training, proceed to Phase 1 with obstacles
python phase_one_obstacles.py

# Continue through all phases...
python phase_two.py
python phase_two_obstacles.py
python phase_three.py
python phase_three_obstacles.py
```

Each training script will:
1. Load the model from the previous phase (if available)
2. Ask how many simulations to run
3. Ask how often to display progress
4. Show the final simulation visually
5. Save the trained model and experience buffer

### Testing

Test the trained model on any phase without further learning:

```bash
python test_robot.py
```

The test interface lets you select any phase and watch the robot navigate using its learned policy.

**Note:** `test_robot.py` loads `robot_phase_three_obstacles_model.pth` (the final trained model) for all phases, to demonstrate generalization across difficulty levels. This file is generated by completing the full training pipeline and is not included in the repository — run the training phases sequentially to produce it.

### Training Configuration

The final model was trained through the following simulation counts per phase:

| Phase | Initial Training | Fine-tuning | Total |
|-------|------------------|-------------|-------|
| Phase 1 | 300 | 10 | 310 |
| Phase 1 (obstacles) | 100 | 10 | 110 |
| Phase 2 | 50 | 10 | 60 |
| Phase 2 (obstacles) | 200 | 10 | 210 |
| Phase 3 | 100 | 10 | 110 |
| Phase 3 (obstacles) | 400 | 10 | 410 |

Transfer learning was applied only during the initial training of each phase. The fine-tuning runs used only the phase-specific model and buffer.

### Hyperparameters (vary by phase)

- **Learning rate**: 0.001 - 0.0025
- **Discount factor (γ)**: 0.9 - 0.99
- **Epsilon decay**: 0.9985 - 0.9994
- **Replay buffer**: 2,000 - 20,000 experiences
- **Batch size**: 32
- **Target network update**: Every 75 - 200 steps

## Future Improvements

- [ ] Refactor the code to have a single training file instead of six
- [ ] Progress even further with the phases complexity to see the model limits

## License

MIT