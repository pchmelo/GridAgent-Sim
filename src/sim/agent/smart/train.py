from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sim.agent.smart.gym_environment import HEMSEnvironment
import os
import torch

def train_sac_agent(total_timesteps=100000, save_path=None, use_gpu=True):
    """Train SAC agent for HEMS"""
    
    if save_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, "models")
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    # Check GPU availability
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"{'='*60}")
    print(f"Training SAC agent on: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Saving to: {save_path}")
    print(f"{'='*60}\n")
    
    # Create training environment
    def make_env():
        env = HEMSEnvironment()
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="sac_hems",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Create SAC model with hyperparameters
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',  # Automatic entropy tuning
        policy_kwargs=dict(net_arch=[256, 256]),  # Neural network architecture
        verbose=1,
        tensorboard_log=log_path,
        device=device
    )
    
    print("Starting training...")
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    return model
