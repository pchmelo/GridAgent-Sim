from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from sim.agent.smart.gym_environment import HEMSEnvironment

def train_with_sb3(total_timesteps=100000):
    """Train using Stable-Baselines3 SAC (recommended!)"""
    
    # Create environment
    env = HEMSEnvironment()
    eval_env = HEMSEnvironment()
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='./models/',
        name_prefix='sac_hems'
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/best/',
        log_path='./logs/',
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        tensorboard_log="./tensorboard/"
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    model.save("models/sac_hems_final")
    
    return model