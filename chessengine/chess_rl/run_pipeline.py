# run_pipeline.py
import yaml
import os
import sys
import subprocess # To call the other scripts

# Ensure chess_rl is in the Python path if running from root
sys.path.append('.')

def load_config(config_path='chess_rl/rl_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = 'chess_rl/rl_config.yaml'
    config = load_config(config_path)

    num_cycles = config['pipeline']['num_cycles']
    checkpoint_dir = config['model']['checkpoint_dir']
    replay_buffer_dir = config['self_play']['replay_buffer_dir']

    # Create necessary directories if they don't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(replay_buffer_dir, exist_ok=True)

    print(f"Starting AlphaZero-style Training Pipeline for {num_cycles} cycles.")
    print(f"Config: {config_path}")
    print(f"Checkpoints will be saved in: {checkpoint_dir}")
    print(f"Replay buffer location: {replay_buffer_dir}")

    for cycle in range(1, num_cycles + 1):
        print(f"\n{'='*20} CYCLE {cycle}/{num_cycles} {'='*20}")

        # === Step 1 & 2: Run Self-Play ===
        print(f"\n--- Running Self-Play (Cycle {cycle}) ---")
        self_play_script = os.path.join('chess_rl', 'run', 'self_play_main.py')
        # Ensure PYTHONPATH includes the project root if necessary
        env = os.environ.copy()
        # env['PYTHONPATH'] = f".{os.pathsep}{env.get('PYTHONPATH', '')}" # Add current dir
        process = subprocess.run([sys.executable, self_play_script], check=True, env=env)
        print(f"--- Self-Play Finished (Cycle {cycle}) ---")


        # === Step 3 & 4: Run Training ===
        print(f"\n--- Running Training (Cycle {cycle}) ---")
        train_script = os.path.join('chess_rl', 'run', 'train.py')
        process = subprocess.run([sys.executable, train_script], check=True, env=env)
        print(f"--- Training Finished (Cycle {cycle}) ---")


    print(f"\n{'='*20} PIPELINE FINISHED AFTER {num_cycles} CYCLES {'='*20}")

if __name__ == "__main__":
    main()