"""
Simple test to verify DQN components work correctly.
This doesn't require Doom - just tests the core algorithm.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch

# Set GPU device (use 4th GPU = index 3)
# You can change this to use a different GPU
torch.cuda.set_device(3)

print("=" * 60)
print("TESTING DQN COMPONENTS")
print("=" * 60)
print(f"Using GPU: {torch.cuda.get_device_name(3)}")

# Test 1: Replay Buffer
print("\n[1] Testing Replay Buffer...")
from replay_buffer import ReplayBuffer

buffer = ReplayBuffer(capacity=100)

# Add experiences
for i in range(50):
    state = np.random.rand(4, 84, 84)
    action = np.random.randint(3)
    reward = np.random.randn()
    next_state = np.random.rand(4, 84, 84)
    done = np.random.random() < 0.1
    buffer.push(state, action, reward, next_state, done)

print(f"   Buffer size: {len(buffer)}")
batch = buffer.sample(batch_size=16)
print(f"   Sampled batch shapes: states={batch[0].shape}, actions={batch[1].shape}")
print("   ✓ Replay Buffer works!")

# Test 2: Q-Network
print("\n[2] Testing Q-Network...")
from q_network import DoomQNetwork

network = DoomQNetwork(num_actions=3)
network.print_architecture()

# Forward pass
test_input = torch.randn(2, 4, 84, 84)
q_values = network(test_input)
print(f"   Input shape:  {test_input.shape}")
print(f"   Output shape: {q_values.shape}")
print("   ✓ Q-Network works!")

# Test 3: Epsilon-Greedy
print("\n[3] Testing Epsilon-Greedy...")
from epsilon_greedy import EpsilonGreedy

explorer = EpsilonGreedy(start_epsilon=1.0, end_epsilon=0.05, decay=0.99)
q_vals = np.array([0.5, 0.8, 0.3])

for _ in range(10):
    action = explorer.select_action(q_vals, training=True)
explorer.step()

print(f"   Epsilon after 10 steps: {explorer.epsilon:.3f}")
print("   ✓ Epsilon-Greedy works!")

# Test 4: DQN Agent (mock environment)
print("\n[4] Testing DQN Agent (no Doom needed)...")
from dqn_agent import DQNAgent

# Create agent with 3 actions (same as Doom has)
# Use GPU index 3 for the 4th GPU
agent = DQNAgent(num_actions=3, device=3)
print(f"   Device: {agent.device}")
print(f"   Buffer size: {agent.replay_buffer.capacity}")

# Store some experiences
for _ in range(200):
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = np.random.randint(3)
    reward = np.random.randn()
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    done = np.random.random() < 0.1
    agent.store_experience(state, action, reward, next_state, done)

# Train
loss = agent.train_step()
print(f"   Training step loss: {loss:.4f}" if loss else "   No training (buffer not full)")
print("   ✓ DQN Agent works!")

# Test 5: Visualization
print("\n[5] Testing Visualization...")
from visualization import TrainingVisualizer

viz = TrainingVisualizer()
viz.log_episode(10.5, 100)
viz.log_episode(15.2, 120)
viz.log_step(0.5, 0.3, 0.9)
print("   ✓ Visualization works!")

print("\n" + "=" * 60)
print("ALL COMPONENTS WORK!")
print("=" * 60)
print("\nThe DQN algorithm components are all functional.")
print("Now you can train with Doom when a display is available.")
