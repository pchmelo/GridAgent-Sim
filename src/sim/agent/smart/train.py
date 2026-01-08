import os
import torch
import random
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sim.agent.smart.gym_environment import HEMSEnvironment


class SaveBestModelReplayBufferCallback(BaseCallback):
    """Save replay buffer whenever best model is updated"""
    def __init__(self, save_path, eval_freq, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.last_saved_step = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls != self.last_saved_step:
            replay_buffer_path = os.path.join(self.save_path, "best_model_replay_buffer.pkl")
            self.model.save_replay_buffer(replay_buffer_path)
            self.last_saved_step = self.n_calls
            if self.verbose > 0:
                print(f"Saved replay buffer to {replay_buffer_path}")
        return True


SEASONAL_DATES = {
    "winter": "2025-01-15",
    "spring": "2025-04-15",
    "summer": "2025-07-15",
    "autumn": "2025-10-15",
}

def make_seasonal_env(date):
    def _init():
        env = HEMSEnvironment(date=date)
        env = Monitor(env)
        return env
    return _init


def export_tensorboard_to_csv(log_dir, output_dir):
    """Export TensorBoard scalar data to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, file))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return
    
    print(f"\nExporting TensorBoard data to CSV...")
    print(f"Found {len(event_files)} event file(s)")
    
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        
        tags = ea.Tags()
        
        for tag in tags['scalars']:
            try:
                events = ea.Scalars(tag)
                
                df = pd.DataFrame([
                    {
                        'step': event.step,
                        'value': event.value,
                        'wall_time': event.wall_time
                    }
                    for event in events
                ])
                
                csv_filename = tag.replace('/', '_') + '.csv'
                csv_path = os.path.join(output_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                
                if tag in ['rollout/ep_rew_mean', 'eval/mean_reward']:
                    print(f"Exported: {csv_filename}")
                
            except Exception as e:
                print(f"Error processing {tag}: {e}")
    
    print(f"CSV files saved to: {output_dir}\n")


def train_single_season(season="summer", total_timesteps=100000, save_path=None, use_gpu=True):
    """Train on a single season first to verify the approach works"""
    
    if save_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, "models", f"single_{season}")
    
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    date = SEASONAL_DATES[season]
    print(f"{'='*60}")
    print(f"Training on single season: {season} ({date})")
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Saving to: {save_path}")
    print(f"{'='*60}\n")
    
    env = DummyVecEnv([make_seasonal_env(date)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    
    eval_env = DummyVecEnv([make_seasonal_env(date)])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=2000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=500000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto',
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
        ),
        verbose=1,
        tensorboard_log=log_path,
        device=device
    )
    
    print("Starting training...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )
    
    model.save(os.path.join(save_path, "final_model"))
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    print(f"\n{'='*60}")
    print(f"Single season training complete!")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")
    
    csv_output_dir = os.path.join(save_path, "csv_exports")
    export_tensorboard_to_csv(log_path, csv_output_dir)
    
    env.close()
    eval_env.close()
    
    return model


def train_sac_agent(total_timesteps=200000, save_path=None, use_gpu=True, 
                    n_envs=4, eval_freq=5000, resume_from=None):
    """Train SAC agent on multiple seasons to avoid overfitting"""
    
    if save_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, "models")
    
    os.makedirs(save_path, exist_ok=True)
    log_path = os.path.join(save_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print(f"Training SAC agent on: {device.upper()}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    print(f"Parallel Environments: {n_envs}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Timesteps per season: ~{total_timesteps // 4:,}")

    if resume_from:
        print(f"\nRESUMING FROM: {resume_from}")

    print(f"\nTraining Seasons:")

    for season, date in SEASONAL_DATES.items():
        print(f"  - {season.capitalize()}: {date}")
    
    print(f"\nSaving to: {save_path}")
    print(f"{'='*60}\n")
    
    env_fns = [make_seasonal_env(date) for date in SEASONAL_DATES.values()]
    
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    
    eval_envs = {}
    for season, date in SEASONAL_DATES.items():
        eval_env = DummyVecEnv([make_seasonal_env(date)])
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
        eval_envs[season] = eval_env
    
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000,
        save_path=save_path,
        name_prefix="sac_hems",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    replay_buffer_callback = SaveBestModelReplayBufferCallback(
        save_path=save_path,
        eval_freq=eval_freq,
        verbose=1
    )
    
    eval_callbacks = []
    for season, eval_env in eval_envs.items():
        season_log_path = os.path.join(log_path, f"eval_{season}")
        os.makedirs(season_log_path, exist_ok=True)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_path, f"best_model_{season}"),
            log_path=season_log_path,
            eval_freq=eval_freq // len(SEASONAL_DATES),
            deterministic=True,
            render=False,
            n_eval_episodes=3,
            verbose=0
        )
        eval_callbacks.append(eval_callback)
    
    main_eval_env = DummyVecEnv([make_seasonal_env(random.choice(list(SEASONAL_DATES.values())))])
    main_eval_env = VecNormalize(main_eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.99)
    
    main_eval_callback = EvalCallback(
        main_eval_env,
        best_model_save_path=save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )
    
    callback_list = CallbackList([checkpoint_callback, main_eval_callback, replay_buffer_callback] + eval_callbacks)
    
    if resume_from and os.path.exists(resume_from):
        print(f"Loading model from: {resume_from}")

        model = SAC.load(
            resume_from,
            env=env,
            device=device,
            verbose=1,
            tensorboard_log=log_path
        )

        print(f"Model loaded successfully!")
        print(f"Continuing training for {total_timesteps:,} more steps...")
        
        if "_steps.zip" in resume_from:
            base_name = os.path.basename(resume_from)
            step_number = base_name.replace("sac_hems_", "").replace(".zip", "")
            replay_buffer_filename = f"sac_hems_replay_buffer_{step_number}.pkl"
            replay_buffer_path = os.path.join(os.path.dirname(resume_from), replay_buffer_filename)
        else:
            replay_buffer_path = resume_from.replace(".zip", "_replay_buffer.pkl")
        
        if os.path.exists(replay_buffer_path):
            print(f"Loading replay buffer from: {replay_buffer_path}")
            model.load_replay_buffer(replay_buffer_path)
            print(f"Replay buffer loaded! Buffer size: {model.replay_buffer.size()}")
        else:
            print(f"No replay buffer found at {replay_buffer_path}")
            print(f"Starting with empty replay buffer")
        
        vec_normalize_path = os.path.join(os.path.dirname(resume_from), "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            print(f"Loading VecNormalize from: {vec_normalize_path}")
            env = VecNormalize.load(vec_normalize_path, env)
    else:
        if resume_from:
            print(f"Model not found at {resume_from}")
            print(f"Creating new model from scratch...")

        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=500000,
            learning_starts=5000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=2,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
            ),
            verbose=1,
            tensorboard_log=log_path,
            device=device
        )
    
    print("Starting training...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
        reset_num_timesteps=False if resume_from else True
    )
    
    model.save(os.path.join(save_path, "final_model"))
    env.save(os.path.join(save_path, "vec_normalize.pkl"))
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"\nSeason-specific best models:")

    for season in SEASONAL_DATES.keys():
        print(f"  - {season.capitalize()}: best_model_{season}.zip")

    print(f"{'='*60}")
    
    csv_output_dir = os.path.join(save_path, "csv_exports")
    export_tensorboard_to_csv(log_path, csv_output_dir)
    
    env.close()
    main_eval_env.close()
    for eval_env in eval_envs.values():
        eval_env.close()
    
    return model
