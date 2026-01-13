"""
================================================================================
01_REPLAY_BUFFER.PY - Understanding Experience Replay
================================================================================

CONCEPT: Why do we need a replay buffer?

In reinforcement learning, an agent learns from sequences of experiences:
    (state, action, reward, next_state, done)

The problem with learning sequentially:
- Consecutive experiences are highly correlated
- The agent only learns from what it just experienced
- This leads to unstable training (catastrophic forgetting)

The solution: Experience Replay
- Store many experiences in a buffer
- Sample random batches to learn from
- This breaks temporal correlations
- Allows efficient reuse of experiences

This is like how humans learn:
- We don't learn immediately from one event
- We reflect on many past experiences
- We learn patterns, not just single events

LEARNING EXERCISES:
1. Add a method to visualize buffer contents over time
2. Track which states appear most frequently
3. Experiment: What happens with different buffer sizes?
4. Question: Why do we sample randomly instead of sequentially?
================================================================================
"""

from collections import deque
import random
import numpy as np
from typing import Tuple, List, Any


class ReplayBuffer:
    """
    A circular buffer that stores experiences for experience replay.

    Key properties:
    - Fixed capacity: when full, oldest experiences are replaced
    - Random sampling: breaks correlation between experiences
    - Efficient storage: uses deque for O(1) append/pop

    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
    """

    def __init__(self, capacity: int):
        """
        Initialize replay buffer with fixed capacity.

        Args:
            capacity: Maximum number of experiences to store
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """
        Store an experience tuple.

        Args:
            state: Current game state (frame)
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray,
                                               np.ndarray]:
        """
        Randomly sample a batch of experiences.

        Why random sampling?
        - Sequential sampling would keep correlated experiences together
        - Random sampling breaks temporal correlations
        - The network learns from diverse experiences

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of arrays: (states, actions, rewards, next_states, dones)
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Batch size {batch_size} > buffer size {len(self.buffer)}"
            )

        batch = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def sample_with_indices(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, List[int]
    ]:
        """
        Sample batch and return indices - useful for priority replay.

        This is a preview of Prioritized Experience Replay (PER):
        - Not all experiences are equally valuable
        - We want to sample important experiences more often
        - Need indices to update priorities
        """
        indices = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices

    def __len__(self) -> int:
        """Return current number of experiences in buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= batch_size

    def get_statistics(self) -> dict:
        """Get buffer statistics - useful for monitoring."""
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "fullness": len(self.buffer) / self.capacity * 100,
        }

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.buffer.clear()

    def get_recent_experiences(self, n: int) -> List[Tuple]:
        """
        Get the n most recent experiences.

        Useful for debugging and understanding recent learning.
        """
        return list(self.buffer)[-n:]


def demo_replay_buffer():
    """
    Demonstration of replay buffer behavior.

    Run this to see:
    - How the buffer fills up
    - How sampling works
    - What happens when capacity is exceeded
    """
    print("=" * 60)
    print("REPLAY BUFFER DEMONSTRATION")
    print("=" * 60)

    # Create a small buffer for demonstration
    buffer = ReplayBuffer(capacity=5)

    print("\n1. Adding experiences...")
    for i in range(7):
        # Simulated experience: (state, action, reward, next_state, done)
        experience = (
            np.random.rand(4, 84, 84),  # Random frame
            i % 3,                       # Action 0, 1, or 2
            np.random.randn(),          # Random reward
            np.random.rand(4, 84, 84),  # Next frame
            i == 6                      # Done on last experience
        )
        buffer.push(*experience)
        stats = buffer.get_statistics()
        print(f"   Added exp {i+1}: buffer size={stats['size']}/{stats['capacity']}")

    print(f"\n2. Buffer is full (capacity={buffer.capacity})")
    print("   Oldest experiences are automatically removed when new ones are added.")

    print("\n3. Sampling a batch...")
    batch = buffer.sample(batch_size=3)
    states, actions, rewards, next_states, dones = batch
    print(f"   Sampled {len(actions)} experiences")
    print(f"   Actions: {actions}")
    print(f"   Rewards: {rewards.round(2)}")
    print(f"   Dones: {dones}")

    print("\n4. Buffer statistics:")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n5. Understanding capacity behavior:")
    print("   - With capacity=5, we can only store 5 experiences")
    print("   - When we add the 6th, the 1st is removed")
    print("   - This is a 'circular' or 'ring' buffer")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: Experience replay breaks temporal correlations")
    print("by sampling randomly from past experiences, not sequentially.")
    print("=" * 60)


if __name__ == "__main__":
    demo_replay_buffer()
