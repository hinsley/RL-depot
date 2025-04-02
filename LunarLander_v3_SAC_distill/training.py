# Note: You must place a teacher.zip file in this directory constituting the
# SAC-trained model from StableBaselines3.
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from torch.distributions import Normal
import random
from tqdm import tqdm

STUDENT_HIDDEN_SIZES = [32, 32, 32, 32]
LEARNING_RATE = 1e-3 # 3e-4
TRAINING_EXAMPLES = 3e5
EPOCHS = 3e3
BATCH_SIZE = 2**12
EVAL_INTERVAL = 3e2
EVAL_EPISODES = 2**5
SAVE_PATH = "LunarLander_v3_SAC_distill"

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=STUDENT_HIDDEN_SIZES):
        super(ActorNetwork, self).__init__()
        
        # Build the layers dynamically based on hidden_sizes
        layers = []
        prev_size = input_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
            
        self.hidden_layers = nn.Sequential(*layers)
        
        # Mean and log_std outputs for continuous actions
        self.mean = nn.Linear(prev_size, output_dim)
        self.log_std = nn.Linear(prev_size, output_dim)
        
        # Log_std bounds
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, x):
        x = self.hidden_layers(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Use reparameterization trick
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        
        # Squash using tanh for bounded actions
        y_t = torch.tanh(x_t)
        
        # For calculating log probabilities correctly with tanh squashing
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean, std
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean).squeeze(0).detach().numpy()
        
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        return y_t.squeeze(0).detach().numpy()

def evaluate_policy(env, policy, eval_episodes=10, deterministic=True):
    """Evaluate policy performance across multiple episodes."""
    avg_reward = 0.0
    
    for _ in range(eval_episodes):
        # Handle different return formats from reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
            
        done = False
        while not done:
            action = policy.get_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            avg_reward += reward
            done = terminated or truncated
            
    avg_reward /= eval_episodes
    return avg_reward

def evaluate_sb3_policy(env, model, eval_episodes=10):
    """Evaluate SB3 policy performance across multiple episodes."""
    avg_reward = 0.0
    
    for _ in range(eval_episodes):
        # Handle different return formats from reset
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
            
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            avg_reward += reward
            done = terminated or truncated
            
    avg_reward /= eval_episodes
    return avg_reward

def kl_divergence_loss(teacher_mean, teacher_std, student_mean, student_std):
    """Compute KL divergence between two Gaussian distributions."""
    kl_div = (torch.log(teacher_std/student_std) + 
               (student_std.pow(2) + (student_mean - teacher_mean).pow(2)) / 
               (2 * teacher_std.pow(2)) - 0.5)
    return kl_div.sum(dim=1).mean()

def save_teacher_samples(states, means, stds, file_path):
    """Save teacher samples to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savez(file_path, 
             states=states, 
             means=means, 
             stds=stds)
    print(f"Teacher samples saved to {file_path}.")

def load_teacher_samples(file_path):
    """Load teacher samples from disk."""
    data = np.load(file_path)
    print(f"Teacher samples loaded from {file_path}.")
    return data['states'], data['means'], data['stds']

def collect_teacher_samples(env, model, num_samples=10000):
    """Collect state-action pairs from the teacher model."""
    print("Starting collect_teacher_samples")
    states = []
    means = []
    stds = []
    
    # For SB3 models, we should use the model's native get_env() method
    sb3_env = model.get_env()
    
    # Get device from model
    device = next(model.policy.parameters()).device
    print(f"Model is on device: {device}")
    
    # Reset the environment (VecEnv always returns just the observation)
    obs = sb3_env.reset()
    
    for _ in tqdm(range(num_samples), desc="Collecting teacher samples"):
        states.append(obs[0])  # VecEnv returns observations as arrays
        
        # Get the action distribution from the SAC policy directly
        with torch.no_grad():
            # Convert to tensor and move to correct device
            obs_tensor = torch.FloatTensor(obs).to(device)
            
            # Get the action parameters
            mean_actions, log_std, _ = model.policy.actor.get_action_dist_params(obs_tensor)
            
            # Move back to CPU for storage
            means.append(mean_actions[0].cpu().numpy())
            stds.append(torch.exp(log_std)[0].cpu().numpy())
        
        # Take an action to advance the environment
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = sb3_env.step(action)
        
        if len(states) >= num_samples:
            break
    
    return np.array(states), np.array(means), np.array(stds)

def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create environment
    env = gym.make("LunarLander-v3", continuous=True)
    eval_env = gym.make("LunarLander-v3", continuous=True)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Load the teacher model
    teacher_model = SAC.load(args.teacher_model_path, env=env)
    
    # Important: when loading a model, SB3 wraps the environment, so create a new one for evaluation
    teacher_eval_env = gym.make("LunarLander-v3", continuous=True)
    
    # Initialize student network
    student_policy = ActorNetwork(
        state_dim, 
        action_dim,
        hidden_sizes=args.hidden_sizes
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(student_policy.parameters(), lr=args.learning_rate)
    
    # Resume training if checkpoint exists and requested
    start_epoch = 0
    if args.resume and os.path.exists(args.student_model_path):
        checkpoint = torch.load(args.student_model_path)
        student_policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}.")
    
    # Check if teacher samples exist on disk, otherwise collect them
    samples_path = args.samples_path
    if os.path.exists(samples_path):
        # Load samples from disk
        states, teacher_means, teacher_stds = load_teacher_samples(samples_path)
    else:
        # Collect samples from teacher and save to disk
        states, teacher_means, teacher_stds = collect_teacher_samples(
            env, teacher_model, num_samples=args.num_samples
        )
        save_teacher_samples(states, teacher_means, teacher_stds, samples_path)
    
    # Convert to torch tensors
    states_tensor = torch.FloatTensor(states)
    teacher_means_tensor = torch.FloatTensor(teacher_means)
    teacher_stds_tensor = torch.FloatTensor(teacher_stds)
    
    # Initial evaluation before training
    print("\n=== Initial Evaluation ===")
    teacher_reward = evaluate_sb3_policy(teacher_eval_env, teacher_model, args.eval_episodes)
    student_reward = evaluate_policy(eval_env, student_policy, args.eval_episodes)
    print(f"Teacher performance: {teacher_reward:.2f}")
    print(f"Student performance: {student_reward:.2f}")
    print(f"Performance ratio: {student_reward/teacher_reward:.2%}")
    
    # Training loop
    batch_size = args.batch_size
    num_batches = len(states) // batch_size
    best_loss = float('inf')  # Initialize best loss as infinity
    
    # Create progress bar for epochs
    progress_bar = tqdm(range(start_epoch, args.epochs), desc=f"Training (Best epoch loss: {best_loss:.4f})")
    
    for epoch in progress_bar:
        total_loss = 0
        
        # Shuffle data
        indices = np.arange(len(states))
        np.random.shuffle(indices)
        
        for batch_idx in range(num_batches):
            batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            batch_states = states_tensor[batch_indices]
            batch_teacher_means = teacher_means_tensor[batch_indices]
            batch_teacher_stds = teacher_stds_tensor[batch_indices]
            
            # Forward pass
            student_means, student_log_stds = student_policy(batch_states)
            student_stds = torch.exp(student_log_stds)
            
            # Compute KL divergence loss
            loss = kl_divergence_loss(
                batch_teacher_means, 
                batch_teacher_stds, 
                student_means, 
                student_stds
            )
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches
        
        # Update best loss if current loss is better
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Update progress bar description with new best loss
            progress_bar.set_description(f"Training (Best loss: {best_loss:.4f})")
        
        # Evaluate periodically
        if (epoch + 1) % args.eval_interval == 0:
            teacher_reward = evaluate_sb3_policy(teacher_eval_env, teacher_model, args.eval_episodes)
            student_reward = evaluate_policy(eval_env, student_policy, args.eval_episodes)
            
            print(f"Teacher performance: {teacher_reward:.2f}")
            print(f"Student performance: {student_reward:.2f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = args.student_model_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
    
    # Final evaluation
    teacher_reward = evaluate_sb3_policy(teacher_eval_env, teacher_model, args.eval_episodes * 2)
    student_reward = evaluate_policy(eval_env, student_policy, args.eval_episodes * 2)
    
    print("\n=== Final Evaluation ===")
    print(f"Teacher average reward: {teacher_reward:.2f}")
    print(f"Student average reward: {student_reward:.2f}")
    print(f"Performance ratio: {student_reward/teacher_reward:.2%}")
    
    # Save final model
    torch.save(student_policy.state_dict(), args.student_final_model_path)
    print(f"Final model saved to {args.student_final_model_path}")

def watch_student_policy(model_path=None):
    """Continuously run episodes with the student policy and render them."""
    # If no model path is provided, use the default path
    if model_path is None:
        model_path = os.path.join(SAVE_PATH, "student_final.pt")
    
    # Create environment with rendering
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    
    # Get state dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Initialize student network
    student_policy = ActorNetwork(state_dim, action_dim)
    
    # Load the trained model
    student_policy.load_state_dict(torch.load(model_path))
    student_policy.eval()
    
    print(f"Watching trained student policy from: {model_path}")
    print("Press Ctrl+C to stop.")
    
    try:
        episode = 0
        while True:
            episode += 1
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action from policy
                action = student_policy.get_action(obs, deterministic=True)
                
                # Execute action
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            print(f"Episode {episode} finished with reward: {episode_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        env.close()

parser = argparse.ArgumentParser(description="SAC Policy Distillation")
parser.add_argument("--teacher_model_path", type=str, default=os.path.join(SAVE_PATH, "teacher"), 
                    help="Path to the teacher model")
parser.add_argument("--student_model_path", type=str, default=os.path.join(SAVE_PATH, "student_checkpoint.pt"), 
                    help="Path to save/load student model checkpoints")
parser.add_argument("--student_final_model_path", type=str, default=os.path.join(SAVE_PATH, "student_final.pt"), 
                    help="Path to save the final student model")
parser.add_argument("--hidden_sizes", type=int, nargs='+', default=STUDENT_HIDDEN_SIZES, 
                    help="Hidden layer sizes for the student policy")
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                    help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=int(BATCH_SIZE),
                    help="Batch size for training")
parser.add_argument("--epochs", type=int, default=int(EPOCHS),
                    help="Number of epochs to train")
parser.add_argument("--eval_interval", type=int, default=int(EVAL_INTERVAL), 
                    help="Interval to evaluate the policy")
parser.add_argument("--eval_episodes", type=int, default=int(EVAL_EPISODES),
                    help="Number of episodes for evaluation")
parser.add_argument("--save_interval", type=int, default=10, 
                    help="Interval to save checkpoints")
parser.add_argument("--num_samples", type=int, default=int(TRAINING_EXAMPLES),
                    help="Number of samples to collect from teacher")
parser.add_argument("--samples_path", type=str, default=os.path.join(SAVE_PATH, "teacher_samples.npz"), 
                    help="Path to save/load teacher samples")
parser.add_argument("--resume", action="store_true", default=True, 
                    help="Resume training from checkpoint if it exists")
parser.add_argument("--no_resume", action="store_false", dest="resume",
                    help="Start training from scratch even if checkpoint exists")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed")

args = parser.parse_args()
main(args)

# Uncomment the line below to run the visualization directly
# watch_student_policy()