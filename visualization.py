"""
================================================================================
06_VISUALIZATION.PY - See the Learning Happen
================================================================================

CONCEPT: Why Visualization Matters

Training a neural network is like flying blind without instrumentation.
You need to see:

1. REWARD CURVE: Is the agent learning to get more reward?
2. LOSS CURVE: Is the network learning (loss decreasing)?
3. Q-VALUES: Is the agent's understanding improving?
4. EXPLORATION: Is epsilon decaying properly?

These metrics tell you:
- If training is working
- If something is wrong
- When to stop training
- What hyperparameters to adjust

VISUALIZATION AS DEBUGGING:
- Loss exploding? Learning rate too high!
- Reward not increasing? Algorithm bug or poor hyperparameters
- Q-values saturating? Network not learning new things

LEARNING EXERCISES:
1. Plot multiple training runs to compare
2. Save plots to TensorBoard for better viewing
3. Create custom metrics for your specific task
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
from pathlib import Path
import json
from datetime import datetime


class TrainingVisualizer:
    """
    Tools to visualize and understand training progress.

    Tracks and plots:
    - Episode rewards
    - Training loss
    - Q-values
    - Epsilon (exploration rate)
    - Episode lengths

    Example:
        >>> viz = TrainingVisualizer()
        >>> for episode in range(500):
        ...     # ... training ...
        ...     viz.log_episode(total_reward, episode_length)
        ...     viz.log_step(loss, q_value, epsilon)
        >>> viz.plot_learning_curves()
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the visualizer.

        Args:
            log_dir: Directory to save plots and logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training metrics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_count = 0

        # Step-level metrics
        self.loss_history: List[float] = []
        self.q_value_history: List[float] = []
        self.epsilon_history: List[float] = []

        # Timing
        self.episode_times: List[float] = []
        self.training_start_time: Optional[datetime] = None

    def on_training_start(self) -> None:
        """Called when training starts."""
        self.training_start_time = datetime.now()

    def log_episode(self, reward: float, length: int,
                    time_taken: Optional[float] = None) -> None:
        """
        Log episode completion.

        Args:
            reward: Total reward for this episode
            length: Number of steps in episode
            time_taken: Time taken for episode (seconds)
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_count += 1

        if time_taken is not None:
            self.episode_times.append(time_taken)

    def log_step(self, loss: Optional[float], q_value: float,
                 epsilon: float, reward: float = 0) -> None:
        """
        Log training step.

        Args:
            loss: Loss from this step (None if no training)
            q_value: Mean Q-value from this step
            epsilon: Current exploration rate
            reward: Immediate reward
        """
        if loss is not None:
            self.loss_history.append(loss)
        self.q_value_history.append(q_value)
        self.epsilon_history.append(epsilon)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log custom metrics.

        Args:
            metrics: Dictionary of metric_name -> value
        """
        for name, value in metrics.items():
            if not hasattr(self, f"custom_{name}"):
                setattr(self, f"custom_{name}", [])
            getattr(self, f"custom_{name}").append(value)

    def plot_learning_curves(self, save_path: Optional[str] = None,
                              show: bool = True) -> plt.Figure:
        """
        Create 4-panel visualization of training progress.

        Panels:
        1. Episode rewards over time
        2. Training loss over time
        3. Average Q-values over time
        4. Epsilon decay over time
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(self.episode_rewards, alpha=0.6, label="Raw")
        if len(self.episode_rewards) > 10:
            # Moving average
            window = min(50, len(self.episode_rewards) // 4)
            moving_avg = self._moving_average(self.episode_rewards, window)
            ax1.plot(moving_avg, label=f"MA-{window}", linewidth=2)
        ax1.set_title("Episode Rewards", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Total Reward")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Training Loss
        ax2 = axes[0, 1]
        if self.loss_history:
            ax2.plot(self.loss_history, alpha=0.6)
            if len(self.loss_history) > 100:
                window = min(100, len(self.loss_history) // 4)
                moving_avg = self._moving_average(self.loss_history, window)
                ax2.plot(moving_avg, label=f"MA-{window}", linewidth=2)
            ax2.set_title("Training Loss", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Training Step")
            ax2.set_ylabel("Loss (Huber)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Q-Values
        ax3 = axes[1, 0]
        if self.q_value_history:
            ax3.plot(self.q_value_history, alpha=0.6)
            if len(self.q_value_history) > 100:
                window = min(100, len(self.q_value_history) // 4)
                moving_avg = self._moving_average(self.q_value_history, window)
                ax3.plot(moving_avg, label=f"MA-{window}", linewidth=2)
            ax3.set_title("Average Q-Values", fontsize=12, fontweight="bold")
            ax3.set_xlabel("Training Step")
            ax3.set_ylabel("Mean Q-Value")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Epsilon Decay
        ax4 = axes[1, 1]
        ax4.plot(self.epsilon_history)
        ax4.set_title("Exploration Rate (Epsilon)", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Epsilon")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def plot_rewards_detailed(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create detailed reward analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Reward distribution
        ax1 = axes[0, 0]
        ax1.hist(self.episode_rewards, bins=50, edgecolor="black", alpha=0.7)
        ax1.axvline(np.mean(self.episode_rewards), color="red", linestyle="--",
                    label=f"Mean: {np.mean(self.episode_rewards):.2f}")
        ax1.set_title("Reward Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Reward")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative rewards
        ax2 = axes[0, 1]
        cumulative = np.cumsum(self.episode_rewards)
        ax2.plot(cumulative)
        ax2.set_title("Cumulative Rewards", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Total Reward")
        ax2.grid(True, alpha=0.3)

        # Episode lengths
        ax3 = axes[1, 0]
        ax3.plot(self.episode_lengths, alpha=0.6)
        if len(self.episode_lengths) > 10:
            window = min(50, len(self.episode_lengths) // 4)
            moving_avg = self._moving_average(self.episode_lengths, window)
            ax3.plot(moving_avg, label=f"MA-{window}", linewidth=2)
        ax3.set_title("Episode Lengths", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Steps")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Reward vs Episode Length
        ax4 = axes[1, 1]
        ax4.scatter(self.episode_lengths, self.episode_rewards, alpha=0.3, s=10)
        ax4.set_title("Reward vs Episode Length", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Episode Length")
        ax4.set_ylabel("Reward")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()
        return fig

    def print_training_summary(self, episode: int) -> str:
        """
        Print a readable training summary.

        Returns:
            Summary string
        """
        recent_rewards = self.episode_rewards[-10:]
        recent_lengths = self.episode_lengths[-10:]

        avg_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_q = np.mean(self.q_value_history[-100:]) if self.q_value_history else 0
        current_epsilon = self.epsilon_history[-1] if self.epsilon_history else 1.0

        summary = (
            f"Episode {episode:4d} | "
            f"Reward: {avg_reward:7.2f} ± {std_reward:5.2f} | "
            f"Length: {avg_length:5.1f} | "
            f"Q: {avg_q:6.3f} | "
            f"ε: {current_epsilon:.3f}"
        )

        print(summary)
        return summary

    def get_statistics(self) -> dict:
        """Get comprehensive training statistics."""
        return {
            "total_episodes": self.episode_count,
            "total_steps": sum(self.episode_lengths),
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0,
            "max_reward": np.max(self.episode_rewards) if self.episode_rewards else 0,
            "min_reward": np.min(self.episode_rewards) if self.episode_rewards else 0,
            "mean_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0,
            "final_epsilon": self.epsilon_history[-1] if self.epsilon_history else 1.0,
            "mean_loss": np.mean(self.loss_history) if self.loss_history else 0,
            "mean_q_value": np.mean(self.q_value_history) if self.q_value_history else 0,
        }

    def save_logs(self, filename: str = "training_logs.json") -> None:
        """Save all logs to JSON file."""
        logs = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "loss_history": self.loss_history,
            "q_value_history": self.q_value_history,
            "epsilon_history": self.epsilon_history,
            "statistics": self.get_statistics(),
        }

        save_path = self.log_dir / filename
        with open(save_path, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"Logs saved to {save_path}")

    def load_logs(self, filepath: str) -> None:
        """Load logs from JSON file."""
        with open(filepath, "r") as f:
            logs = json.load(f)

        self.episode_rewards = logs.get("episode_rewards", [])
        self.episode_lengths = logs.get("episode_lengths", [])
        self.loss_history = logs.get("loss_history", [])
        self.q_value_history = logs.get("q_value_history", [])
        self.epsilon_history = logs.get("epsilon_history", [])

        print(f"Logs loaded from {filepath}")

    @staticmethod
    def _moving_average(data: List[float], window: int) -> np.ndarray:
        """Compute moving average."""
        data = np.array(data)
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode="valid")


class TrainingLogger:
    """
    Simple CSV-based logger for integration with external tools.
    """

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._write_header()

    def _write_header(self) -> None:
        """Write CSV header."""
        with open(self.log_file, "w") as f:
            f.write("episode,reward,length,mean_q,mean_loss,epsilon,time\n")

    def log(self, episode: int, reward: float, length: int,
            mean_q: float, mean_loss: float, epsilon: float,
            time_taken: float) -> None:
        """Log a training episode."""
        with open(self.log_file, "a") as f:
            f.write(f"{episode},{reward},{length},{mean_q},{mean_loss},{epsilon},{time_taken}\n")


def demo_visualization():
    """Demonstrate visualization tools."""
    print("=" * 60)
    print("VISUALIZATION TOOLS DEMONSTRATION")
    print("=" * 60)

    # Create visualizer
    viz = TrainingVisualizer(log_dir="demo_logs")

    # Simulate training data
    print("\n1. Generating simulated training data...")
    for episode in range(500):
        # Simulate reward curve (improving over time)
        base_reward = -50 + episode * 0.5
        reward = base_reward + np.random.randn() * 20
        reward = np.clip(reward, -100, 150)

        length = np.random.randint(50, 200)
        viz.log_episode(reward, length)

        # Simulate step-level metrics
        loss = np.exp(-episode / 200) + np.random.randn() * 0.1
        q_value = 0.1 + episode * 0.002 + np.random.randn() * 0.05
        epsilon = max(0.05, 1.0 * 0.995 ** episode)
        viz.log_step(loss, q_value, epsilon)

        if (episode + 1) % 100 == 0:
            viz.print_training_summary(episode + 1)

    print("\n2. Generating plots...")
    viz.plot_learning_curves(save_path="demo_logs/learning_curves.png")

    print("\n3. Generating detailed reward analysis...")
    viz.plot_rewards_detailed(save_path="demo_logs/reward_analysis.png")

    print("\n4. Statistics:")
    stats = viz.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}")

    print("\n5. Saving logs...")
    viz.save_logs()

    print("\n" + "=" * 60)
    print("KEY INSIGHT: Visualization is essential for understanding")
    print("training dynamics. The reward curve shows if the agent is")
    print("learning, while the loss curve shows if the network is fitting.")
    print("=" * 60)


if __name__ == "__main__":
    demo_visualization()
