import os
# Force CPU usage by setting environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gymnasium
import torch  # Import torch to configure CPU usage
torch.set_num_threads(1)  # Optional: limit CPU threads
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Whether to train, or if not, to just evaluate.
TRAIN = True

# Hyperparameters.
# Training parameters.
TOTAL_TIMESTEPS = 3000000
SAVE_FREQUENCY = 10000
EVAL_FREQUENCY = 20000

# PPO Agent parameters.
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_RANGE = 0.2
TARGET_KL = 0.01
N_EPOCHS = 10
BATCH_SIZE = 256
GAE_LAMBDA = 0.95

# Network architecture.
POLICY_HIDDEN_SIZES = [32, 32, 32, 32]
VALUE_HIDDEN_SIZES = [256, 256, 256]

# Training function.
def train_ppo(env_name, total_timesteps=TOTAL_TIMESTEPS, save_path="LunarLander_v3_PPO/saved_model"):
    # Create the environment.
    env = gymnasium.make(env_name)
    
    # Set up the model.
    policy_kwargs = dict(
        net_arch=dict(
            pi=POLICY_HIDDEN_SIZES,
            vf=VALUE_HIDDEN_SIZES
        )
    )
    
    # Check if we have a saved model to resume from
    final_model_path = os.path.join(save_path, "final_model")
    if os.path.exists(f"{final_model_path}.zip"):
        print(f"Resuming training from {final_model_path}...")
        model = PPO.load(final_model_path, env=env)
    else:
        print("Starting training from scratch...")
        # Create PPO agent.
        model = PPO(
            "MlpPolicy", 
            env,
            learning_rate=LEARNING_RATE,
            n_steps=2048,  # Default PPO buffer size per update.
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            target_kl=TARGET_KL,
            policy_kwargs=policy_kwargs,
            device="cpu",  # Force CPU usage
            verbose=1
        )
    
    # Set up saving callbacks.
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQUENCY,
        save_path=save_path,
        name_prefix="ppo_lunarlander"
    )
    
    # Train the agent.
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save the final model.
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    
    return model

# Visualize a trained agent.
def visualize_agent(env_name, model):
    env = gymnasium.make(env_name, render_mode="human")
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    
    while not (done or truncated):
        # Select action using the model.
        action, _ = model.predict(obs, deterministic=True)
        
        # Take a step in the environment.
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        # Render.
        env.render()
        # time.sleep(0.01)
    
    print(f"Visualization episode finished with total reward: {total_reward:.2f}")
    env.close()

def estimate_network_memory(policy_hidden_sizes=POLICY_HIDDEN_SIZES, 
                           value_hidden_sizes=VALUE_HIDDEN_SIZES,
                           obs_dim=8,  # LunarLander-v3 observation dimension
                           action_dim=4):  # LunarLander-v3 action dimension
    """
    Estimate memory requirements for policy and value networks based on 
    network architecture, without loading any model files.
    
    Args:
        policy_hidden_sizes (list): Hidden layer sizes for policy network.
        value_hidden_sizes (list): Hidden layer sizes for value network.
        obs_dim (int): Observation space dimension.
        action_dim (int): Action space dimension.
        
    Returns:
        dict: Dictionary containing parameter counts and memory requirements.
    """
    # Calculate policy network parameters.
    policy_params = 0
    prev_dim = obs_dim
    
    # Add parameters for each layer in policy network.
    for dim in policy_hidden_sizes:
        # Weight matrix + bias vector.
        policy_params += (prev_dim * dim) + dim
        prev_dim = dim
    
    # Output layer from last hidden to action space.
    policy_params += (prev_dim * action_dim) + action_dim
    
    # Calculate value network parameters.
    value_params = 0
    prev_dim = obs_dim
    
    # Add parameters for each layer in value network.
    for dim in value_hidden_sizes:
        # Weight matrix + bias vector.
        value_params += (prev_dim * dim) + dim
        prev_dim = dim
    
    # Output layer (scalar value).
    value_params += prev_dim + 1  # weights + bias for single output
    
    # Total parameters.
    total_params = policy_params + value_params
    
    # Calculate memory size (assuming float32 - 4 bytes per parameter).
    bytes_per_param = 4  # float32
    policy_memory_bytes = policy_params * bytes_per_param
    value_memory_bytes = value_params * bytes_per_param
    total_memory_bytes = total_params * bytes_per_param
    
    # Format output sizes.
    def format_size(bytes):
        kb = bytes / 1024
        mb = kb / 1024
        if mb >= 1:
            return f"{mb:.2f} MB"
        elif kb >= 1:
            return f"{kb:.2f} KB"
        else:
            return f"{bytes} bytes"
    
    # Print results.
    print("\nEstimated Model Memory Requirements:")
    print("=" * 50)
    print(f"Policy Network Architecture: {[obs_dim]} + {policy_hidden_sizes} + [{action_dim}]")
    print(f"  - Parameters: {policy_params:,}")
    print(f"  - Memory size: {format_size(policy_memory_bytes)}")
    
    print(f"\nValue Network Architecture: {[obs_dim]} + {value_hidden_sizes} + [1]")
    print(f"  - Parameters: {value_params:,}")
    print(f"  - Memory size: {format_size(value_memory_bytes)}")
    
    print("\nTotal:")
    print(f"  - Parameters: {total_params:,}")
    print(f"  - Memory size: {format_size(total_memory_bytes)}")
    print("=" * 50)
    
    return {
        "policy_params": policy_params,
        "value_params": value_params,
        "policy_memory": policy_memory_bytes,
        "value_memory": value_memory_bytes,
        "total_params": total_params,
        "total_memory": total_memory_bytes
    }

# Main execution.
if __name__ == "__main__":
    # Estimate memory requirements based on network architecture.
    estimate_network_memory()
    
    if TRAIN:
        # Train the agent.
        model = train_ppo("LunarLander-v3", TOTAL_TIMESTEPS)
        
        # Evaluate the trained agent.
        eval_env = gymnasium.make("LunarLander-v3")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Visualize the trained agent.
        visualize_agent("LunarLander-v3", model)
    else:
        # Load the trained model.
        model_path = "LunarLander_v3_PPO/saved_model/final_model"
        model = PPO.load(model_path)
        
        # Evaluate the trained agent.
        eval_env = gymnasium.make("LunarLander-v3")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Visualize the trained agent in a loop.
        print("Visualizing agent indefinitely. Press Ctrl+C to stop.")
        while True:
            visualize_agent("LunarLander-v3", model)
            print("Starting new visualization episode...")