import os

import gymnasium as gym
from stable_baselines3 import SAC

# Network architecture.
POLICY_HIDDEN_SIZES = [128, 128, 128]
VALUE_HIDDEN_SIZES = [256, 256, 256]

# Model save path.
SAVE_PATH = "LunarLander_v3_SAC/saved_model"

env_train = gym.make(
  "LunarLander-v3",
  render_mode=None,
  continuous=True
)
env_test = gym.make(
  "LunarLander-v3",
  render_mode="human",
  continuous=True
)

policy_kwargs = dict(
  net_arch=dict(
    pi = POLICY_HIDDEN_SIZES,
    qf = VALUE_HIDDEN_SIZES
  )
)

# Create new model.
# model = SAC("MlpPolicy", env_train, verbose=1, policy_kwargs=policy_kwargs)

# Load existing model.
model = SAC.load(os.path.join(SAVE_PATH, "sac_llv3"), env_train)

# Train and save the updated model.
model.learn(total_timesteps=1e5, log_interval=4)
model.save(os.path.join(SAVE_PATH, "sac_llv3"))

obs, info = env_test.reset()
while True:
  action, _states = model.predict(obs, deterministic=True)
  obs, reward, terminated, truncated, info = env_test.step(action)
  if terminated or truncated:
    obs, info = env_test.reset()