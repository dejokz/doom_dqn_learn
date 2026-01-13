import sys
import os
from pathlib import Path
from collections import deque
from typing import Tuple, Optional, Dict
import numpy as np
import cv2

# --- REMOVED: os.environ["SDL_VIDEODRIVER"] = "dummy" ---
# (We are using xvfb, so we want the standard X11 driver, not dummy)

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

try:
    import vizdoom as vzd
except ImportError:
    print("Error: vizdoom not installed.")
    sys.exit(1)

class DoomGame:
    def __init__(self, scenario: str = "defend_the_center"):
        self.game = vzd.DoomGame()

        # Load scenario configuration
        # (Load config FIRST so we can override settings later)
        scenarios = {
            "basic": "basic.cfg",
            "defend_the_center": "defend_the_center.cfg",
        }
        scenario_path = os.path.join(vzd.scenarios_path, scenarios[scenario])
        self.game.load_config(scenario_path)

        # --- CRITICAL FIXES FOR VM/HEADLESS ---
        self.game.set_window_visible(False) 
        self.game.set_sound_enabled(False)   # <--- FIXES 0x8 CRASH
        self.game.set_render_hud(False)      # Optimization
        # --------------------------------------

        # Resolution settings
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24) 

        # Actions: [Turn Left, Turn Right, Shoot]
        self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        self.game.init()
        self.frame_stack = 4
        self.frames = deque(maxlen=4)

    def reset(self):
        self.game.new_episode()
        self.frames.clear()
        frame = self._get_frame()
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        return self._get_observation(), {}

    def step(self, action: int):
        self.game.set_action(self.actions[action])
        self.game.advance_action(4)
        
        frame = self._get_frame()
        self.frames.append(frame)
        
        reward = self.game.get_last_reward()
        done = self.game.is_episode_finished()
        
        return self._get_observation(), reward, done, False, {}

    def _get_frame(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None: return np.zeros((84, 84), dtype=np.float32)
        
        frame = state.screen_buffer
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
        return frame.astype(np.float32) / 255.0

    def get_display_frame(self):
        state = self.game.get_state()
        if state is None: return None
        return cv2.cvtColor(state.screen_buffer, cv2.COLOR_RGB2BGR)

    def _get_observation(self):
        return np.stack(list(self.frames), axis=0)

    def close(self):
        self.game.close()

def demo_with_recording():
    print("=" * 60)
    print("RUNNING HEADLESS ON VM - RECORDING VIDEO")
    print("=" * 60)

    env = DoomGame(scenario="defend_the_center")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('gameplay_demo.avi', fourcc, 30.0, (320, 240))
    print("Recording to 'gameplay_demo.avi'...")

    for episode in range(2):
        env.reset()
        step = 0
        total_reward = 0
        while True:
            raw_frame = env.get_display_frame()
            if raw_frame is not None:
                out.write(raw_frame)

            action = np.random.randint(3)
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            if done: break
        
        print(f"Episode {episode+1}: Steps: {step}, Reward: {total_reward}")

    out.release()
    env.close()
    print("\nDone! Download 'gameplay_demo.avi'.")

if __name__ == "__main__":
    demo_with_recording()