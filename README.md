# DQN Learning Journey: Train Your Own Doom Agent

Learn Deep Q-Networks from scratch by building a Doom-playing agent. This educational project teaches you every component of DQN with extensive documentation and visualization.

## What You'll Learn

- **Experience Replay**: How agents learn from past experiences
- **Q-Networks**: Neural networks that predict action values
- **Exploration vs Exploitation**: Balancing random and optimal actions
- **Target Networks**: Why stable learning requires two networks
- **Frame Preprocessing**: Preparing game frames for neural networks
- **Training Dynamics**: Understanding loss, rewards, and Q-values

## Quick Start

```bash
# 1. Install system dependencies for ViZDoom (Linux)
sudo apt-get install build-essential cmake libboost-all-dev libsdl2-dev libswscale-dev libasmutils0

# 2. Create virtual environment and install Python dependencies
cd doom_dqn_learn
uv venv
source .venv/bin/activate
uv pip install gymnasium torch numpy opencv-python matplotlib tqdm vizdoom

# 3. Quick test (3 episodes, human render)
python 07_train_step_by_step.py --quick-test

# 4. Full training
python 07_train_step_by_step.py --episodes 500

# 5. Evaluate trained agent
python 09_evaluate_and_play.py --checkpoint checkpoints/dqn_final.pt --episodes 10
```

## Project Structure

```
doom_dqn_learn/
├── replay_buffer.py          # Experience replay implementation
├── q_network.py              # CNN for Q-value prediction
├── dqn_agent.py              # Core DQN algorithm
├── env_wrapper.py            # Frame preprocessing
├── epsilon_greedy.py         # Exploration strategy
├── visualization.py          # Training plots
├── 07_train_step_by_step.py  # Training script
├── 09_evaluate_and_play.py   # Testing script
└── README.md                 # This file
```

## Learning Path

### Phase 1: Understanding Components (Days 1-3)

1. **replay_buffer.py** - Learn how experience replay breaks temporal correlations
2. **q_network.py** - Understand CNN architecture for image processing
3. **epsilon_greedy.py** - Master the exploration-exploitation tradeoff
4. **env_wrapper.py** - See how frames are preprocessed

### Phase 2: The Algorithm (Days 4-6)

5. **dqn_agent.py** - Combine all components into the DQN algorithm
6. Understand the Q-learning update and target network

### Phase 3: Training (Days 7-10)

7. **07_train_step_by_step.py** - Train with extensive logging
8. Watch learning curves and understand training dynamics

### Phase 4: Evaluation (Days 11-14)

9. **09_evaluate_and_play.py** - Test and analyze your agent
10. Compare to random baseline

### Phase 5: Experiments (Ongoing)

- Remove replay buffer and see training fail
- Try different learning rates
- Implement Double DQN or Dueling DQN
- Experiment with different epsilon schedules

## Key Concepts

### Q-Values

The Q-value Q(s, a) represents the expected total reward if we:
1. Are in state s
2. Take action a
3. Then follow our policy forever

The agent learns to predict these Q-values and chooses actions with the highest predicted value.

### The Bellman Equation

Q(s, a) = r + γ * max_a' Q(s', a')

This is the core of Q-learning:
- The new Q-value = current reward + discounted future reward
- The network learns to approximate this equation

### Experience Replay

Without replay:
- Agent learns sequentially from recent experiences
- Correlated experiences cause unstable learning

With replay:
- Store experiences in a buffer
- Sample random batches
- Breaks correlations, enables efficient reuse

### Target Network

Problem: Both sides of the Bellman equation use the same network
- Creates a "chasing moving target" problem

Solution: Use two networks
- Main network: learns from experiences
- Target network: provides stable targets
- Soft update blends them together

## Watching Training

### Console Output

The training script prints progress every 10 episodes:
```
Episode 100 | Reward: 25.50 ± 15.20 | Length: 125.3 | Q: 0.423 | ε: 0.605
```

- **Reward**: Mean reward over last 10 episodes
- **Length**: Mean episode length
- **Q**: Mean Q-value estimate
- **ε**: Current exploration rate

### Generated Plots

After training, three plots are saved to `logs/`:
- `learning_curves.png`: Rewards, loss, Q-values, epsilon
- `reward_analysis.png`: Reward distribution, cumulative rewards
- `training_logs.json`: Raw data for custom analysis

### TensorBoard (Optional)

```bash
pip install tensorboard
python 07_train_step_by_step.py --tensorboard
tensorboard --logdir logs
```

## Common Issues

### Training Doesn't Converge

- Lower learning rate (try 1e-4)
- Increase buffer size
- Check if gamma is too high (try 0.95)

### Loss Explodes

- Gradient clipping (already implemented)
- Lower learning rate
- Check for NaN in observations

### Agent Does Nothing

- Epsilon decayed too fast
- Reward function issues
- Network not receiving proper input

### ViZDoom Crashes

- Install system dependencies
- Check ViZDoom compatibility
- Try different render mode

## Further Reading

- [Human-level control through deep RL](https://arxiv.org/abs/1312.5602) - Mnih et al. 2015
- [Deep Reinforcement Learning Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On) - Packt
- [HuggingFace RL Course](https://huggingface.co/learn/rl-course)

## Next Steps

1. Try different Doom scenarios (Basic, HealthGathering)
2. Implement improvements (Double DQN, Dueling DQN, PER)
3. Apply to other environments (Atari, custom games)
4. Experiment with hyperparameters
5. Share your trained model!

## License

MIT - Use for learning!
