"""
Ghost Agent using Reinforcement Learning (Stable-Baselines3)
Autonomous agent that learns to chase Pacman
"""

import os
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    # Create dummy classes when sb3 is not available
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0
    EvalCallback = None
    DummyVecEnv = None
    VecNormalize = None
    Monitor = None
    PPO = None
    DQN = None
    A2C = None
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress."""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Mean reward = {mean_reward:.2f}, Mean length = {mean_length:.0f}")
        return True
    
    def _on_rollout_end(self) -> None:
        # Get episode info from the buffer
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])


class GhostAgent:
    """
    Reinforcement Learning agent for Ghost behavior.
    Uses PPO (Proximal Policy Optimization) by default.
    """
    
    ALGORITHMS = {
        'ppo': PPO if HAS_SB3 else None,
        'dqn': DQN if HAS_SB3 else None,
        'a2c': A2C if HAS_SB3 else None
    }
    
    def __init__(
        self,
        algorithm: str = 'ppo',
        model_path: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Initialize the Ghost Agent.
        
        Args:
            algorithm: RL algorithm to use ('ppo', 'dqn', 'a2c')
            model_path: Path to load pre-trained model
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3")
        
        self.algorithm_name = algorithm.lower()
        self.algorithm_class = self.ALGORITHMS.get(self.algorithm_name)
        
        if self.algorithm_class is None:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: {list(self.ALGORITHMS.keys())}")
        
        self.device = device
        self.model = None
        self.env = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def create_model(
        self,
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        policy_kwargs: Optional[Dict] = None,
        verbose: int = 1
    ):
        """
        Create a new RL model.
        
        Args:
            env: Gymnasium environment
            learning_rate: Learning rate
            n_steps: Steps per update (PPO/A2C)
            batch_size: Batch size
            n_epochs: Epochs per update (PPO)
            gamma: Discount factor
            policy_kwargs: Additional policy arguments
            verbose: Verbosity level
        """
        self.env = env
        
        # Default policy architecture
        if policy_kwargs is None:
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
        
        if self.algorithm_name == 'ppo':
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=self.device
            )
        elif self.algorithm_name == 'dqn':
            self.model = DQN(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=self.device
            )
        elif self.algorithm_name == 'a2c':
            self.model = A2C(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=self.device
            )
        
        return self.model
    
    def train(
        self,
        total_timesteps: int = 100000,
        log_freq: int = 1000,
        eval_freq: int = 10000,
        eval_episodes: int = 5,
        save_path: Optional[str] = None,
        save_freq: int = 10000
    ) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps
            log_freq: Logging frequency
            eval_freq: Evaluation frequency
            eval_episodes: Episodes per evaluation
            save_path: Path to save model checkpoints
            save_freq: Checkpoint save frequency
            
        Returns:
            Training statistics
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Setup callbacks
        callbacks = [TrainingCallback(log_freq=log_freq)]
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            # Create evaluation environment
            eval_env = DummyVecEnv([lambda: Monitor(self.env.envs[0].unwrapped.__class__(
                agent_type=self.env.envs[0].unwrapped.agent_type
            ))])
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=eval_episodes,
                deterministic=True
            )
            callbacks.append(eval_callback)
        
        # Train
        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        if save_path:
            final_path = os.path.join(save_path, "final_model")
            self.save(final_path)
            print(f"Final model saved to: {final_path}")
        
        return {
            'total_timesteps': total_timesteps,
            'algorithm': self.algorithm_name
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action for given observation.
        
        Args:
            observation: Current state observation
            deterministic: Use deterministic policy
            
        Returns:
            Action to take
        """
        if self.model is None:
            raise ValueError("Model not loaded or created.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)
    
    def save(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        self.model.save(path)
        print(f"Model saved to: {path}")
    
    def load(self, path: str):
        """Load model from file."""
        if self.algorithm_name == 'ppo':
            self.model = PPO.load(path, device=self.device)
        elif self.algorithm_name == 'dqn':
            self.model = DQN.load(path, device=self.device)
        elif self.algorithm_name == 'a2c':
            self.model = A2C.load(path, device=self.device)
        
        print(f"Model loaded from: {path}")
    
    def get_action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        """
        Get action probabilities for given observation.
        Only works with policy gradient methods (PPO, A2C).
        
        Args:
            observation: Current state observation
            
        Returns:
            Action probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded or created.")
        
        if self.algorithm_name == 'dqn':
            # For DQN, return Q-values normalized as probabilities
            obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
            q_values = self.model.q_net(obs_tensor).detach().cpu().numpy()
            # Softmax
            exp_q = np.exp(q_values - np.max(q_values))
            return exp_q / exp_q.sum()
        else:
            # For PPO/A2C, get action distribution
            obs_tensor = self.model.policy.obs_to_tensor(observation)[0]
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.detach().cpu().numpy()
            return probs[0]


def create_training_env(agent_type: str = 'ghost', n_envs: int = 1):
    """
    Create vectorized training environment.
    
    Args:
        agent_type: 'ghost' or 'pacman'
        n_envs: Number of parallel environments
        
    Returns:
        Vectorized environment
    """
    from environments.pacman_env import PacmanEnv
    
    def make_env():
        env = PacmanEnv(agent_type=agent_type)
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    return env
