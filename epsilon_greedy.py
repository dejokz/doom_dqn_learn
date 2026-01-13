"""
================================================================================
05_EPSILON_GREEDY.PY - The Exploration vs Exploitation Dilemma
================================================================================

CONCEPT: Why do we need exploration?

The fundamental problem in reinforcement learning:
- EXPLOITATION: Use what we know is good (maximize immediate reward)
- EXPLORATION: Try new things to discover better strategies (may find better rewards)

This is like choosing a restaurant:
- Exploitation: Go to your favorite restaurant
- Exploration: Try a new restaurant (might be better, might be worse)

In DQN:
- Exploitation: Always pick the action with highest Q-value
- Exploration: Sometimes pick a random action

WHY RANDOM EXPLORATION?
- The agent doesn't initially know which actions are good
- Q-values start random and converge slowly
- Random exploration ensures the agent discovers all possibilities

THE EPSILON-GREEDY STRATEGY:
- With probability ε: explore (random action)
- With probability 1-ε: exploit (best action)

Usually ε starts high (1.0 = 100% exploration) and decreases over time.
This is called "annealing" - gradually shifting from exploration to exploitation.

LEARNING EXERCISES:
1. Plot epsilon over time for different decay values
2. Visualize: How does exploration rate change during training?
3. Experiment: What if epsilon decays too fast? Too slow?
================================================================================
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


class EpsilonGreedy:
    """
    Epsilon-Greedy exploration strategy.

    Balance between exploration (random actions) and exploitation (best actions).

    Example:
        >>> explorer = EpsilonGreedy(start_epsilon=1.0, end_epsilon=0.05, decay=0.995)
        >>> q_values = np.array([0.5, 0.8, 0.3])  # Predicted Q-values
        >>> action = explorer.select_action(q_values, training=True)
    """

    def __init__(self, start_epsilon: float = 1.0, end_epsilon: float = 0.05,
                 decay: float = 0.995, step_type: str = "episode"):
        """
        Initialize epsilon-greedy explorer.

        Args:
            start_epsilon: Initial exploration rate (1.0 = 100% random)
            end_epsilon: Final exploration rate (0.05 = 5% random)
            decay: Multiplicative decay factor per step (0.995 = 0.5% decrease)
            step_type: "episode" or "step" - what triggers decay
        """
        if not 0 <= end_epsilon <= start_epsilon <= 1.0:
            raise ValueError("Epsilon values must satisfy: 0 <= end <= start <= 1")

        if not 0 < decay <= 1.0:
            raise ValueError("Decay must be in (0, 1]")

        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay = decay
        self.step_type = step_type

        self.epsilon = start_epsilon

    def select_action(self, q_values: np.ndarray,
                      training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            q_values: Q-values for each action [num_actions]
            training: If True, may explore randomly
                     If False, always exploit (greedy)

        Returns:
            Action index to take
        """
        if not training or np.random.random() >= self.epsilon:
            # Exploitation: choose best action
            # Q: Why argmax? A: We want the action with highest Q-value
            return int(np.argmax(q_values))
        else:
            # Exploration: choose random action
            # Q: Why random? A: To discover potentially better actions
            return np.random.randint(len(q_values))

    def select_action_with_probs(self, q_values: np.ndarray,
                                  training: bool = True) -> tuple:
        """
        Select action and return probability distribution used.

        Useful for understanding and visualization.
        """
        n_actions = len(q_values)
        if training and np.random.random() < self.epsilon:
            probs = np.ones(n_actions) / n_actions
            action = np.random.randint(n_actions)
        else:
            probs = np.zeros(n_actions)
            action = int(np.argmax(q_values))
            probs[action] = 1.0

        return action, probs

    def step(self) -> None:
        """
        Decay epsilon after each episode/step.

        The decay formula: ε = max(end_ε, ε × decay)

        Why multiply?
        - Linear decay: ε_new = ε - constant
        - Exponential decay: ε_new = ε × decay (smoother, more common)

        Why max with end_epsilon?
        - We don't want epsilon to go below end_epsilon
        - This maintains some exploration forever (5% is common)
        """
        self.epsilon = max(self.end_epsilon, self.epsilon * self.decay)

    def reset(self) -> None:
        """Reset epsilon to initial value (for new training runs)."""
        self.epsilon = self.start_epsilon

    def get_epsilon(self) -> float:
        """Current epsilon value."""
        return self.epsilon

    def decay_schedule(self, n_steps: int) -> np.ndarray:
        """
        Generate the epsilon schedule for n_steps.

        Useful for plotting and analysis.
        """
        schedule = []
        eps = self.start_epsilon
        for _ in range(n_steps):
            schedule.append(eps)
            eps = max(self.end_epsilon, eps * self.decay)
        return np.array(schedule)

    def __repr__(self) -> str:
        return (f"EpsilonGreedy(start={self.start_epsilon:.2f}, "
                f"end={self.end_epsilon:.2f}, decay={self.decay:.4f})")


class LinearEpsilonGreedy:
    """
    Alternative: Linear epsilon decay instead of exponential.

    ε(t) = start_ε - (t / total_steps) × (start_ε - end_ε)

    Some papers use linear decay, some use exponential.
    Exponential is more common because it's smoother.
    """

    def __init__(self, start_epsilon: float = 1.0, end_epsilon: float = 0.05,
                 total_steps: int = 1000, step_type: str = "episode"):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.total_steps = total_steps
        self.step_type = step_type
        self.current_step = 0

    def select_action(self, q_values: np.ndarray,
                      training: bool = True) -> int:
        if not training or np.random.random() >= self.epsilon:
            return int(np.argmax(q_values))
        else:
            return np.random.randint(len(q_values))

    @property
    def epsilon(self) -> float:
        """Compute current epsilon based on linear decay."""
        progress = min(self.current_step / self.total_steps, 1.0)
        return self.start_epsilon - progress * (self.start_epsilon - self.end_epsilon)

    def step(self) -> None:
        self.current_step += 1

    def reset(self) -> None:
        self.current_step = 0


def demo_epsilon_greedy():
    """Demonstrate epsilon-greedy exploration."""
    print("=" * 60)
    print("EPSILON-GREEDY EXPLORATION DEMONSTRATION")
    print("=" * 60)

    # Create explorer
    explorer = EpsilonGreedy(start_epsilon=1.0, end_epsilon=0.05, decay=0.995)

    print("\n1. Initial state (100% exploration):")
    q_values = np.array([0.5, 0.8, 0.3])
    for _ in range(5):
        action, probs = explorer.select_action_with_probs(q_values, training=True)
        print(f"   Action: {action}, Probabilities: {probs.round(3)}")

    print("\n2. After 100 steps of decay:")
    for _ in range(100):
        explorer.step()
    print(f"   Epsilon: {explorer.epsilon:.4f}")
    print(f"   (Agent now exploits {100-explorer.epsilon*100:.1f}% of the time)")

    print("\n3. After 500 steps:")
    for _ in range(400):
        explorer.step()
    print(f"   Epsilon: {explorer.epsilon:.4f}")

    print("\n4. Comparison of decay schedules:")
    decays = [0.99, 0.995, 0.999]
    n_steps = 500

    plt.figure(figsize=(10, 5))
    for decay in decays:
        exp = EpsilonGreedy(start_epsilon=1.0, end_epsilon=0.01, decay=decay)
        schedule = exp.decay_schedule(n_steps)
        plt.plot(schedule, label=f"decay={decay}")

    plt.xlabel("Steps")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("epsilon_decay.png", dpi=100)
    plt.show()
    print("   Saved plot to epsilon_decay.png")

    print("\n5. Key observations:")
    print("   - decay=0.99: Fast decay, agent exploits quickly")
    print("   - decay=0.999: Slow decay, agent explores longer")
    print("   - There's a tradeoff: too fast = poor exploration,")
    print("     too slow = slow convergence")

    print("\n" + "=" * 60)
    print("KEY INSIGHT: Exploration is essential because the agent")
    print("starts knowing nothing. Epsilon-greedy provides a simple,")
    print("effective way to balance exploration and exploitation.")
    print("=" * 60)


class DecayingSchedule:
    """
    Advanced: Custom exploration schedules.

    Sometimes we want more complex schedules:
    - Start fast, then slow down
    - Keep exploring in promising regions
    - Use uncertainty estimates

    This is an area of active research!
    """

    def __init__(self, schedule_type: str = "exponential", **kwargs):
        """
        Args:
            schedule_type: "exponential", "linear", "step", "polynomial"
            **kwargs: Schedule-specific parameters
        """
        self.schedule_type = schedule_type
        self.kwargs = kwargs

    def get_epsilon(self, step: int, total_steps: int) -> float:
        """Get epsilon for a given step."""
        if self.schedule_type == "exponential":
            decay = self.kwargs.get("decay", 0.995)
            start = self.kwargs.get("start_epsilon", 1.0)
            end = self.kwargs.get("end_epsilon", 0.05)
            eps = start * (decay ** step)
            return max(end, eps)

        elif self.schedule_type == "linear":
            start = self.kwargs.get("start_epsilon", 1.0)
            end = self.kwargs.get("end_epsilon", 0.05)
            progress = min(step / total_steps, 1.0)
            return start - progress * (start - end)

        return 1.0


if __name__ == "__main__":
    demo_epsilon_greedy()
