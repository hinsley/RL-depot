import gymnasium
import os
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
POLICY_HIDDEN_SIZES = [128, 128, 128]
VALUE_HIDDEN_SIZES = [64, 64, 64]

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

# Main execution.
if __name__ == "__main__":
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