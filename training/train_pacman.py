#!/usr/bin/env python3
"""
Training script for Pacman RL Agent
Trains an autonomous Pacman agent to collect dots and avoid ghosts
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.pacman_env import PacmanEnv
from agents.ghost_agent import GhostAgent, create_training_env


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_pacman_agent(config: dict, save_path: str = 'models/pacman_agent'):
    """
    Train the Pacman RL agent.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save trained model
    """
    print("=" * 60)
    print("Pacman Agent Training")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env_config = config.get('environment', {})
    env = create_training_env(
        agent_type='pacman',
        n_envs=env_config.get('n_envs', 4)
    )
    print(f"   - Created {env_config.get('n_envs', 4)} parallel environments")
    
    # Create agent (reuse GhostAgent class with pacman env)
    print("\n2. Creating agent...")
    agent_config = config.get('agent', {})
    agent = GhostAgent(
        algorithm=agent_config.get('algorithm', 'ppo'),
        device=agent_config.get('device', 'auto')
    )
    
    # Create model
    policy_kwargs = agent_config.get('policy_kwargs', {})
    agent.create_model(
        env,
        learning_rate=agent_config.get('learning_rate', 3e-4),
        n_steps=agent_config.get('n_steps', 2048),
        batch_size=agent_config.get('batch_size', 64),
        n_epochs=agent_config.get('n_epochs', 10),
        gamma=agent_config.get('gamma', 0.99),
        policy_kwargs=policy_kwargs
    )
    print(f"   - Algorithm: {agent_config.get('algorithm', 'ppo').upper()}")
    print(f"   - Learning rate: {agent_config.get('learning_rate', 3e-4)}")
    
    # Train
    print("\n3. Starting training...")
    training_config = config.get('training', {})
    
    os.makedirs(save_path, exist_ok=True)
    
    stats = agent.train(
        total_timesteps=training_config.get('total_timesteps', 100000),
        log_freq=training_config.get('log_freq', 1000),
        eval_freq=training_config.get('eval_freq', 10000),
        eval_episodes=training_config.get('eval_episodes', 5),
        save_path=save_path
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total timesteps: {stats['total_timesteps']}")
    print(f"Model saved to: {save_path}")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='Train Pacman RL Agent')
    parser.add_argument('--config', type=str, default='training/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save-path', type=str, default='models/pacman_agent',
                       help='Path to save trained model')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Override total training timesteps')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    if os.path.exists(args.config):
        config_path = args.config
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Override to pacman agent type
    config['environment']['agent_type'] = 'pacman'
    
    # Override timesteps if provided
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps
    
    # Train pacman agent
    agent = train_pacman_agent(config, args.save_path)
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
