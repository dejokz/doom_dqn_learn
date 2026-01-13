"""
================================================================================
07_TRAIN_STEP_BY_STEP.PY - Watch the Agent Learn
================================================================================

This script provides a step-by-step training loop with detailed logging.
UPDATED: Compatible with the fixed headless env_wrapper.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import time
from typing import Optional

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# --- UPDATE: Import DoomGame directly ---
from env_wrapper import DoomGame 
from dqn_agent import DQNAgent
from visualization import TrainingVisualizer


def train_with_logging(
    # Environment settings
    env_id: str = "VizdoomDefendTheCenter-v0",
    frame_stack: int = 4,
    render_mode: Optional[str] = None,

    # Training settings
    num_episodes: int = 500,
    max_steps_per_episode: int = 1000,

    # Agent settings
    learning_rate: float = 2.5e-4,
    buffer_size: int = 10000,
    batch_size: int = 64,
    gamma: float = 0.99,
    tau: float = 0.005,
    learning_starts: int = 1000,

    # Exploration settings
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,

    # Logging settings
    log_interval: int = 10,
    save_interval: int = 50,
    checkpoint_dir: str = "checkpoints",
    logs_dir: str = "logs",
) -> tuple:
    
    # Create directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DQN TRAINING - Watch the agent learn!")
    print("=" * 70)

    print("\n[1] Setting up environment...")
    
    # --- UPDATE: Instantiating DoomGame directly ---
    # We ignore render_mode and frame_stack here because the fixed env_wrapper 
    # handles them internally to ensure stability on the VM.
    env = DoomGame(scenario="defend_the_center")

    # --- UPDATE: Getting action count directly ---
    num_actions = len(env.actions)
    
    print(f"    Environment: defend_the_center")
    print(f"    Number of actions: {num_actions}")
    print(f"    Frame stack: {env.frame_stack}")

    print("\n[2] Setting up DQN agent...")
    agent = DQNAgent(
        num_actions=num_actions,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        learning_starts=learning_starts,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )

    print(f"    Learning rate: {learning_rate}")
    print(f"    Buffer size: {buffer_size}")
    print(f"    Batch size: {batch_size}")

    print("\n[3] Setting up visualization...")
    visualizer = TrainingVisualizer(log_dir=logs_dir)
    visualizer.on_training_start()

    print("\n[4] Starting training...")
    print("-" * 70)

    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        # Reset environment
        state, _ = env.reset()

        total_reward = 0
        episode_loss = []
        step = 0

        while step < max_steps_per_episode:
            # Select action
            action = agent.select_action(state, training=True)

            # Take action
            # Note: The fixed wrapper returns 5 values (gymnasium style)
            next_state, reward, done, truncated, _ = env.step(action)

            total_reward += reward

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            state = next_state
            step += 1

            if done or truncated:
                break

        # Episode finished
        time_taken = time.time() - episode_start_time
        agent.update_exploration()
        visualizer.log_episode(total_reward, step, time_taken)

        if episode_loss:
            avg_loss = np.mean(episode_loss)
        else:
            avg_loss = None

        # Get Q-value estimate for logging
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_value = agent.q_network(state_tensor).mean().item()

        visualizer.log_step(avg_loss, q_value, agent.explorer.epsilon)

        if episode % log_interval == 0:
            visualizer.print_training_summary(episode)

        if episode % save_interval == 0:
            checkpoint_path = f"{checkpoint_dir}/dqn_episode_{episode}.pt"
            agent.save_checkpoint(checkpoint_path)

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print(f"Total episodes: {num_episodes}")

    stats = visualizer.get_statistics()
    print("\nFinal Statistics:")
    print(f"  Mean reward: {stats['mean_reward']:.2f}")
    print(f"  Final epsilon: {stats['final_epsilon']:.3f}")

    agent.save_checkpoint(f"{checkpoint_dir}/dqn_final.pt")

    print("\nGenerating plots...")
    visualizer.plot_learning_curves(save_path=f"{logs_dir}/learning_curves.png")
    visualizer.save_logs()

    env.close()
    return agent, visualizer


def quick_test(num_episodes: int = 10):
    """Quick test to verify everything works."""
    print("=" * 70)
    print("QUICK TEST - Verifying all components work")
    print("=" * 70)

    # --- UPDATE: Use DoomGame direct class ---
    env = DoomGame(scenario="defend_the_center")
    
    # --- UPDATE: Get actions length directly ---
    agent = DQNAgent(num_actions=len(env.actions))

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.store_experience(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train_step()

            state = next_state
            if done: break

        agent.update_exploration()
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Epsilon = {agent.explorer.epsilon:.3f}")

    print("\nQuick test complete!")
    env.close()


def run_with_tensorboard(log_dir: str = "logs"):
    from torch.utils.tensorboard import SummaryWriter

    print("=" * 70)
    print("TRAINING WITH TENSORBOARD")
    print("=" * 70)
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # --- UPDATE: Use DoomGame direct class ---
    env = DoomGame(scenario="defend_the_center")
    agent = DQNAgent(num_actions=len(env.actions))

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            agent.store_experience(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            if done: break

        agent.update_exploration()
        writer.add_scalar("Reward/Episode", total_reward, episode)
        
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

    writer.close()
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    if args.quick_test:
        quick_test(num_episodes=3)
    elif args.tensorboard:
        run_with_tensorboard()
    else:
        train_with_logging(num_episodes=args.episodes)