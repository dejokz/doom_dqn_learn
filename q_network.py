"""
================================================================================
02_Q_NETWORK.PY - Building the Neural Network
================================================================================

CONCEPT: The Q-Network as a Function Approximator

What is a Q-function?
- Q(s, a) = expected total reward from state s if we take action a
- If we know Q(s, a) for all actions, we just pick the action with highest Q!

The problem: The state space is huge
- Doom screen: 84 x 84 pixels, with 4 channels (stacked frames)
- States: (84*84*4) = 28,224 dimensions
- We can't tabulate Q(s, a) for every state!

The solution: Function Approximation
- Use a neural network to approximate Q(s, a)
- Network takes state, outputs Q-values for all actions
- This is called a "Deep Q-Network" (DQN)

Why CNN (Convolutional Neural Network)?
- Images have spatial structure
- CNNs learn to detect patterns (edges, textures, objects)
- Much more efficient than fully connected layers for images

ARCHITECTURE (from Mnih et al. 2015):
    Input:  [batch, 4, 84, 84]  - 4 stacked grayscale frames
    Conv1:  32 filters, 8x8, stride 4 -> ReLU
    Conv2:  64 filters, 4x4, stride 2 -> ReLU
    Conv3:  64 filters, 3x3, stride 1 -> ReLU
    FC1:    512 units -> ReLU
    FC2:    num_actions units (no activation!)

Why no activation on output?
- Q-values can be any real number (positive or negative)
- We need raw Q-values to compare actions
- Using softmax or sigmoid would restrict the range

LEARNING EXERCISES:
1. Print the model architecture - understand each layer's output shape
2. Forward pass a single frame and see the Q-values
3. Modify architecture: add/remove layers, change filter sizes
4. Question: Why no activation on final layer?
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DoomQNetwork(nn.Module):
    """
    Convolutional Neural Network for Doom Q-Learning.

    This network takes game frames and outputs Q-values for each action.
    The Q-value represents "expected total reward if I take this action here."

    Architecture follows the original DQN paper (Mnih et al. 2015):
    - 3 convolutional layers for feature extraction
    - 2 fully connected layers for action selection

    Example:
        >>> network = DoomQNetwork(num_actions=3)
        >>> state = torch.randn(1, 4, 84, 84)  # Batch of 1
        >>> q_values = network(state)  # [1, 3]
        >>> best_action = q_values.argmax(dim=1).item()
    """

    def __init__(self, num_actions: int, in_channels: int = 4, device: str = "auto"):
        """
        Initialize the Q-Network.

        Args:
            num_actions: Number of possible actions in the environment
            in_channels: Number of input frames (default 4 for frame stacking)
            device: Device to run on ("auto", "cuda:3", etc.)
        """
        super().__init__()

        self.num_actions = num_actions
        self.in_channels = in_channels

        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device(device)

        print(f"   Q-Network device: {self.device}")

        # ============================================================
        # CONVOLUTIONAL LAYERS - Extract visual features
        # ============================================================

        # Conv1: First layer - learns low-level features (edges, textures)
        # Input:  [batch, 4, 84, 84]
        # Output: [batch, 32, 20, 20]  (84-8)/4 + 1 = 20.5 -> 20
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=8,
            stride=4
        )

        # Conv2: Second layer - learns higher-level patterns
        # Input:  [batch, 32, 20, 20]
        # Output: [batch, 64, 9, 9]  (20-4)/2 + 1 = 9
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )

        # Conv3: Third layer - learns complex features
        # Input:  [batch, 64, 9, 9]
        # Output: [batch, 64, 7, 7]  (9-3)/1 + 1 = 7
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )

        # ============================================================
        # FULLY CONNECTED LAYERS - Map features to Q-values
        # ============================================================

        # FC1: First fully connected layer
        # Input: [batch, 64 * 7 * 7] = [batch, 3136]
        # Output: [batch, 512]
        self.fc1 = nn.Linear(64 * 7 * 7, 512)

        # FC2: Output layer - Q-value for each action
        # Input: [batch, 512]
        # Output: [batch, num_actions]
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: game frame -> Q-values.

        Args:
            x: Input tensor of shape [batch, 4, 84, 84]

        Returns:
            Q-values tensor of shape [batch, num_actions]

        THE FORWARD PASS STEP BY STEP:
        1. Apply conv1 + ReLU
        2. Apply conv2 + ReLU
        3. Apply conv3 + ReLU
        4. Flatten (reshape to [batch, features])
        5. Apply fc1 + ReLU
        6. Apply fc2 (NO activation - we need raw Q-values!)
        """
        # Step 1-3: Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Step 4: Flatten for fully connected layers
        # Preserve batch dimension, flatten everything else
        x = x.view(x.size(0), -1)

        # Step 5-6: Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # NO activation - raw Q-values

        return x

    def get_action(self, state: torch.Tensor, deterministic: bool = True) -> int:
        """
        Get the best action from Q-values.

        Args:
            state: Input tensor [1, 4, 84, 84]
            deterministic: If True, always pick best action (exploit)
                         If False, use epsilon-greedy (handled elsewhere)

        Returns:
            Action index with highest Q-value
        """
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax(dim=1).item()

    def print_architecture(self) -> None:
        """Print detailed architecture information."""
        print("=" * 60)
        print("DOOM Q-NETWORK ARCHITECTURE")
        print("=" * 60)
        print(f"Input shape:     [batch, {self.in_channels}, 84, 84]")
        print(f"Output shape:    [batch, {self.num_actions}]")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print("-" * 60)
        print("Layer-by-layer shapes:")
        print(f"  Input:        [{self.in_channels}, 84, 84]")
        print(f"  Conv1:        [32, 20, 20]  (8x8 kernel, stride 4)")
        print(f"  Conv2:        [64, 9, 9]    (4x4 kernel, stride 2)")
        print(f"  Conv3:        [64, 7, 7]    (3x3 kernel, stride 1)")
        print(f"  Flatten:      [3136]")
        print(f"  FC1:          [512]")
        print(f"  FC2:          [{self.num_actions}] (Q-values per action)")
        print("=" * 60)
        print("\nNOTE: Final layer has NO activation function!")
        print("This is because Q-values can be any real number.")
        print("=" * 60)

    def count_parameters(self) -> dict:
        """Count parameters by layer."""
        return {
            "conv1": self.conv1.weight.numel(),
            "conv2": self.conv2.weight.numel(),
            "conv3": self.conv3.weight.numel(),
            "fc1": self.fc1.weight.numel(),
            "fc2": self.fc2.weight.numel(),
            "total": sum(p.numel() for p in self.parameters()),
        }


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture.

    KEY INSIGHT: Sometimes the state value is independent of the action.
    Example: In Doom, just surviving might be valuable regardless of aiming.

    Dueling DQN splits the network into:
    - Value stream: V(s) - how good is this state overall?
    - Advantage stream: A(s, a) - how much better is each action?

    Q(s, a) = V(s) + A(s, a) - mean(A(s,))

    This allows the network to learn state value separately from action advantages.
    """

    def __init__(self, num_actions: int, in_channels: int = 4):
        super().__init__()

        # Shared convolutional features (same as DQN)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Value stream: estimates V(s)
        self.value_fc = nn.Linear(64 * 7 * 7, 512)
        self.value_out = nn.Linear(512, 1)

        # Advantage stream: estimates A(s, a)
        self.advantage_fc = nn.Linear(64 * 7 * 7, 512)
        self.advantage_out = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared convolutional features
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Value stream
        v = F.relu(self.value_fc(x))
        v = self.value_out(v)  # [batch, 1]

        # Advantage stream
        a = F.relu(self.advantage_fc(x))
        a = self.advantage_out(a)  # [batch, num_actions]

        # Combine: Q = V + A - mean(A)
        # Subtract mean to make advantages centered around zero
        return v + a - a.mean(dim=1, keepdim=True)


def demo_q_network():
    """Demonstrate the Q-network."""
    print("=" * 60)
    print("Q-NETWORK DEMONSTRATION")
    print("=" * 60)

    # Create network for 3 actions
    network = DoomQNetwork(num_actions=3)

    # Print architecture
    network.print_architecture()

    # Print parameter counts
    counts = network.count_parameters()
    print("\nParameter counts:")
    for layer, count in counts.items():
        print(f"  {layer}: {count:,}")

    # Forward pass with random data
    print("\n1. Forward pass demonstration:")
    batch_size = 2
    state = torch.randn(batch_size, 4, 84, 84)
    q_values = network(state)
    print(f"   Input shape:  {state.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   Q-values for batch[0]: {q_values[0].detach().numpy().round(2)}")
    print(f"   Best action for batch[0]: {q_values[0].argmax().item()}")

    # Demonstrate gradient computation
    print("\n2. Gradient computation (required for learning):")
    loss = q_values.mean()
    loss.backward()
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed for all {len(list(network.parameters()))} layers")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: The network maps high-dimensional images")
    print("to low-dimensional action values. This is function approximation!")
    print("=" * 60)


if __name__ == "__main__":
    demo_q_network()
