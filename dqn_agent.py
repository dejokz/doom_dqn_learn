"""
================================================================================
03_DQN_AGENT.PY - The Core DQN Algorithm
================================================================================

CONCEPT: How DQN Actually Learns

The DQN algorithm has several key components that work together:

1. Q-NETWORK: Predicts Q-values for each action in a state
2. TARGET NETWORK: Provides stable Q-value targets for training
3. REPLAY BUFFER: Stores experiences for off-policy learning
4. EPSILON-GREEDY: Balances exploration and exploitation

THE Q-LEARNING UPDATE:
    Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q_target(s',a') - Q(s,a)]
================================================================================
"""

import sys
from pathlib import Path

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Tuple, List

from replay_buffer import ReplayBuffer
from q_network import DoomQNetwork
from epsilon_greedy import EpsilonGreedy


class DQNAgent:
    """
    Deep Q-Network Agent.

    This agent learns to play Doom by:
    1. Observing game states
    2. Storing experiences in a replay buffer
    3. Learning from random batches of experiences
    4. Balancing exploration and exploitation

    Key hyperparameters:
    - learning_rate: How fast the network learns
    - buffer_size: How many experiences to remember
    - batch_size: How many experiences to learn from at once
    - gamma: How much to discount future rewards
    - tau: How fast the target network updates

    Example:
        >>> agent = DQNAgent(num_actions=3)
        >>> state = np.random.rand(4, 84, 84)
        >>> action = agent.select_action(state)
        >>> agent.store_experience(state, action, reward, next_state, done)
        >>> loss = agent.train_step()
    """

    def __init__(
        self,
        num_actions: int,
        # Q-Network parameters
        learning_rate: float = 2.5e-4,
        in_channels: int = 4,
        # Replay buffer parameters
        buffer_size: int = 10000,
        batch_size: int = 64,
        # Learning parameters
        gamma: float = 0.99,  # Discount factor
        # Target network parameters
        tau: float = 0.005,  # Soft update rate (lower = more stable)
        # Exploration parameters
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        # Training parameters
        learning_starts: int = 1000,  # Don't train until buffer has enough data
        train_freq: int = 1,  # Train every N steps
        # Device
        device: str = "auto",
    ):
        """
        Initialize the DQN agent.

        All hyperparameters are explained in the class docstring.
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.step_count = 0

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            # Allow specifying GPU index directly
            self.device = torch.device(f"cuda:{device}")
        elif device.startswith("cuda:"):
            # Allow specifying specific GPU like "cuda:3"
            self.device = torch.device(device)
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # ============================================================
        # Q-Networks
        # ============================================================

        # Main network: learns from experiences
        self.q_network = DoomQNetwork(
            num_actions=num_actions,
            in_channels=in_channels
        ).to(self.device)

        # Target network: provides stable Q-targets
        self.target_network = DoomQNetwork(
            num_actions=num_actions,
            in_channels=in_channels
        ).to(self.device)

        # Initialize target network to match main network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Freeze target network (we only backprop through main network)
        for param in self.target_network.parameters():
            param.requires_grad = False

        # ============================================================
        # Optimizer
        # ============================================================

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )

        # ============================================================
        # Replay Buffer
        # ============================================================

        self.replay_buffer = ReplayBuffer(buffer_size)

        # ============================================================
        # Exploration Strategy
        # ============================================================

        self.explorer = EpsilonGreedy(
            start_epsilon=epsilon_start,
            end_epsilon=epsilon_end,
            decay=epsilon_decay
        )

        # Training metrics
        self.training_steps = 0
        self.loss_history: List[float] = []
        self.q_value_history: List[float] = []

    def select_action(self, state: np.ndarray,
                      training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state [4, 84, 84]
            training: If True, may explore randomly

        Returns:
            Action index to take
        """
        if not training:
            # Evaluation mode: always exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor).squeeze(0)
                return int(q_values.argmax().cpu().numpy())

        # Training mode: epsilon-greedy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().numpy()
            return self.explorer.select_action(q_values, training=True)

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """
        Store an experience in the replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        This is the core DQN learning algorithm:

        1. Sample a batch of experiences from replay buffer
        2. Compute current Q-values from main network
        3. Compute target Q-values from target network
        4. Compute loss: MSE between current and target
        5. Backpropagate and update main network
        6. Soft update target network

        Returns:
            Loss value (None if not enough data)
        """
        self.step_count += 1

        # Don't train until we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.step_count < self.learning_starts:
            return None

        # Train every N steps
        if self.step_count % self.train_freq != 0:
            return None

        # ============================================================
        # Step 1: Sample batch from replay buffer
        # ============================================================
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ============================================================
        # Step 2: Compute current Q-values
        # ============================================================
        # Q(s, a) for the actions that were actually taken
        current_q = self.q_network(states)  # [batch, num_actions]
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch]

        # ============================================================
        # Step 3: Compute target Q-values
        # ============================================================
        # Q-target = r + γ * max_a' Q_target(s', a')
        # For terminal states: Q-target = r (no future reward)

        with torch.no_grad():
            # Double DQN: use main network to select action, target to evaluate
            # This reduces overestimation of Q-values
            next_q_values = self.q_network(next_states)  # [batch, num_actions]
            next_actions = next_q_values.argmax(dim=1)  # Best action according to main network

            # Get Q-values from target network for those actions
            next_q_target = self.target_network(next_states)  # [batch, num_actions]
            next_q_target = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Compute target
            target_q = rewards + (1 - dones) * self.gamma * next_q_target

        # ============================================================
        # Step 4: Compute loss
        # ============================================================
        # Huber loss is more robust to outliers than MSE
        loss = F.smooth_l1_loss(current_q, target_q)

        # ============================================================
        # Step 5: Backpropagation
        # ============================================================
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients for stability (from Mnih et al. 2015)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)

        self.optimizer.step()

        # ============================================================
        # Step 6: Soft update target network
        # ============================================================
        # θ_target = τ * θ_main + (1 - τ) * θ_target
        self._soft_update_target_network()

        # Record metrics
        self.training_steps += 1
        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q.mean().item())

        return loss.item()

    def _soft_update_target_network(self) -> None:
        """
        Soft update of target network parameters.

        Instead of copying weights directly (hard update),
        we blend main network weights into target network.

        With tau=0.005:
        - Each step: 0.5% of main network goes into target
        - After 1000 steps: ~99% of target is from main
        """
        for main_param, target_param in zip(
            self.q_network.parameters(),
            self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * main_param.data + (1 - self.tau) * target_param.data
            )

    def update_exploration(self) -> None:
        """Decay epsilon after each episode."""
        self.explorer.step()

    def get_statistics(self) -> dict:
        """Get training statistics."""
        return {
            "training_steps": self.training_steps,
            "buffer_size": len(self.replay_buffer),
            "current_epsilon": self.explorer.epsilon,
            "mean_loss": np.mean(self.loss_history[-100:]) if self.loss_history else None,
            "mean_q_value": np.mean(self.q_value_history[-100:]) if self.q_value_history else None,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save agent state to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "explorer_epsilon": self.explorer.epsilon,
            "training_steps": self.training_steps,
            "loss_history": self.loss_history,
            "q_value_history": self.q_value_history,
            "num_actions": self.num_actions,
        }

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load agent state from file."""
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.explorer.epsilon = checkpoint["explorer_epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.loss_history = checkpoint["loss_history"]
        self.q_value_history = checkpoint["q_value_history"]

        print(f"Checkpoint loaded from {path}")


class DoubleDQN(DQNAgent):
    """
    Double DQN: Reduces overestimation bias.

    The key improvement: use main network to select action,
    target network to evaluate it.

    Standard DQN:
        Q_target = r + γ * max_a' Q_target(s', a')

    Double DQN:
        Q_target = r + γ * Q_target(s', argmax_a' Q_main(s', a'))

    This prevents the agent from overestimating Q-values
    by using two different networks for selection and evaluation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_step(self) -> Optional[float]:
        """Double DQN training step."""
        self.step_count += 1

        if len(self.replay_buffer) < self.batch_size:
            return None

        if self.step_count < self.learning_starts:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: select with main, evaluate with target
        with torch.no_grad():
            # Use main network to select best action
            next_q_main = self.q_network(next_states)
            next_actions = next_q_main.argmax(dim=1)

            # Use target network to evaluate
            next_q_target = self.target_network(next_states)
            next_q_target = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + (1 - dones) * self.gamma * next_q_target

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self._soft_update_target_network()

        self.training_steps += 1
        self.loss_history.append(loss.item())
        self.q_value_history.append(current_q.mean().item())

        return loss.item()


def demo_dqn_agent():
    """Demonstrate the DQN agent."""
    print("=" * 60)
    print("DQN AGENT DEMONSTRATION")
    print("=" * 60)

    # Create agent
    agent = DQNAgent(num_actions=3)

    print("\n1. Agent components:")
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n2. Action selection:")
    # Create a mock state
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = agent.select_action(state, training=True)
    print(f"   Selected action: {action} (training mode, epsilon={agent.explorer.epsilon:.3f})")

    action_eval = agent.select_action(state, training=False)
    print(f"   Selected action: {action_eval} (evaluation mode)")

    print("\n3. Storing experience:")
    next_state = np.random.rand(4, 84, 84).astype(np.float32)
    agent.store_experience(state, action, 0.1, next_state, False)
    print(f"   Buffer size: {len(agent.replay_buffer)}")

    print("\n4. Training (will train once buffer has enough data):")
    for _ in range(1500):  # Fill buffer and train
        agent.store_experience(
            np.random.rand(4, 84, 84).astype(np.float32),
            np.random.randint(3),
            np.random.randn(),
            np.random.rand(4, 84, 84).astype(np.float32),
            np.random.random() < 0.1
        )
        loss = agent.train_step()

    if loss is not None:
        print(f"   Training step completed!")
        print(f"   Loss: {loss:.4f}")
        stats = agent.get_statistics()
        print(f"   Training steps: {stats['training_steps']}")
        print(f"   Mean loss (last 100): {np.mean(agent.loss_history[-100:]):.4f}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: The agent learns by minimizing the difference")
    print("between predicted Q-values and target Q-values (Bellman equation).")
    print("The target network stabilizes this learning process.")
    print("=" * 60)


if __name__ == "__main__":
    demo_dqn_agent()
