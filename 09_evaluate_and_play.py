"""
================================================================================
09_EVALUATE_AND_PLAY.PY - Testing Your Trained Agent (VM Compatible)
================================================================================

Run from the doom_dqn_learn directory:
    xvfb-run -a python 09_evaluate_and_play.py --play
    xvfb-run -a python 09_evaluate_and_play.py --benchmark checkpoints/
================================================================================
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional

# Add script directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

# Import the direct class, not the old function
from env_wrapper import DoomGame
from dqn_agent import DQNAgent

def load_weights_correctly(agent, checkpoint_path):
    """Helper function to handle different checkpoint formats."""
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)
    
    # Check for the specific key your agent uses
    if 'q_network_state_dict' in checkpoint:
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    elif 'model_state_dict' in checkpoint:
        agent.q_network.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Assume the file contains only the weights
        agent.q_network.load_state_dict(checkpoint)
        
    agent.q_network.eval()
    return agent

def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 10,
) -> Dict[str, float]:
    """Evaluate agent metrics (Math only, no video)."""
    print("=" * 60)
    print("AGENT EVALUATION")
    print("=" * 60)

    env = DoomGame(scenario="defend_the_center")

    rewards = []
    lengths = []

    print(f"\nRunning {num_episodes} evaluation episodes...")

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0

        while True:
            # Deterministic action (No random exploration)
            action = agent.select_action(state, training=False)
            
            state, reward, done, _, info = env.step(action)
            total_reward += reward
            episode_length += 1
            
            if done: break

        rewards.append(total_reward)
        lengths.append(episode_length)
        print(f"  Episode {episode + 1}: Reward = {total_reward:.2f}, Length = {episode_length}")

    env.close()

    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths),
    }

    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    return metrics

def record_and_analyze(
    checkpoint_path: str,
    output_name: str = "agent_analysis.avi"
):
    """
    Records video AND prints Q-values to console.
    """
    print("=" * 60)
    print("RECORDING AGENT PLAYBACK & Q-VALUES")
    print("=" * 60)

    # Load Agent
    env = DoomGame(scenario="defend_the_center")
    agent = DQNAgent(num_actions=len(env.actions))
    
    # --- FIX: Use robust loading function ---
    load_weights_correctly(agent, checkpoint_path)

    # Setup Video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_name, fourcc, 30.0, (320, 240))
    print(f"Recording video to: {output_name}")

    action_names = ["Left", "Right", "Shoot"]

    for i in range(3): # Record 3 episodes
        state, _ = env.reset()
        done = False
        step = 0
        
        print(f"\n--- Episode {i+1} ---")
        
        while not done:
            # 1. Save Frame
            frame = env.get_display_frame()
            if frame is not None: out.write(frame)

            # 2. Analyze Q-Values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor).squeeze(0).cpu().numpy()
            
            # 3. Take Action
            action = agent.select_action(state, training=False)
            state, reward, done, _, _ = env.step(action)
            
            # Print analysis every 10 steps
            if step % 10 == 0:
                best_act = action_names[np.argmax(q_values)]
                print(f"Step {step:03}: Q={q_values.round(1)} | Choice: {best_act} | Reward: {reward}")

            step += 1
            if step > 500: break

    out.release()
    env.close()
    print(f"\nDone! Download '{output_name}' to watch.")

def benchmark_checkpoints(checkpoint_dir: str):
    """Find the best model in the folder."""
    print("=" * 60)
    print("BENCHMARKING CHECKPOINTS")
    print("=" * 60)
    
    path = Path(checkpoint_dir)
    checkpoints = sorted(path.glob("dqn_episode_*.pt"))
    
    results = []
    
    for cp in checkpoints:
        print(f"Testing {cp.name}...")
        # Load specific checkpoint
        env = DoomGame(scenario="defend_the_center")
        agent = DQNAgent(num_actions=3)
        
        # --- FIX: Use robust loading function ---
        load_weights_correctly(agent, str(cp))
            
        metrics = evaluate_agent(agent, num_episodes=5)
        results.append((cp.name, metrics["mean_reward"]))
        env.close()

    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    # Sort by reward
    results.sort(key=lambda x: x[1], reverse=True)
    for name, score in results:
        print(f"{name:<30} | Score: {score:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dqn_final.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--play", action="store_true", help="Record video with Q-value logs")
    parser.add_argument("--benchmark", type=str, default=None, help="Folder to benchmark")
    
    args = parser.parse_args()

    if args.benchmark:
        benchmark_checkpoints(args.benchmark)
    elif args.play:
        record_and_analyze(args.checkpoint)
    else:
        # Standard numeric evaluation
        env = DoomGame(scenario="defend_the_center")
        agent = DQNAgent(num_actions=3)
        
        # --- FIX: Use robust loading function ---
        load_weights_correctly(agent, args.checkpoint)
            
        evaluate_agent(agent, args.episodes)
        env.close()