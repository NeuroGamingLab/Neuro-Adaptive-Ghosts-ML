#!/usr/bin/env python3
"""
Training script for Ghost RL Agent
Trains an autonomous ghost agent to chase Pacman
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
from unsupervised.state_encoder import StateEncoder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_ghost_agent(config: dict, save_path: str = 'models/ghost_agent'):
    """
    Train the ghost RL agent.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save trained model
    """
    print("=" * 60)
    print("Ghost Agent Training")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env_config = config.get('environment', {})
    env = create_training_env(
        agent_type='ghost',
        n_envs=env_config.get('n_envs', 4)
    )
    print(f"   - Created {env_config.get('n_envs', 4)} parallel environments")
    
    # Create agent
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


def collect_state_data(env, num_episodes: int = 100) -> list:
    """Collect state data for unsupervised learning."""
    print(f"\nCollecting state data from {num_episodes} episodes...")
    
    states = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            states.append(obs[0].copy())  # Get observation from first env
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step([action])
            done = terminated[0] or truncated[0]
        
        if (ep + 1) % 10 == 0:
            print(f"   Episode {ep + 1}/{num_episodes} - Collected {len(states)} states")
    
    return states


def train_state_encoder(states: list, config: dict, save_path: str = 'models/state_encoder.pkl'):
    """Train unsupervised state encoder."""
    print("\n4. Training State Encoder...")
    
    encoder_config = config.get('unsupervised', {}).get('state_encoder', {})
    
    encoder = StateEncoder(
        n_components=encoder_config.get('n_components', 32),
        n_clusters=encoder_config.get('n_clusters', 20),
        use_minibatch=encoder_config.get('use_minibatch', True)
    )
    
    import numpy as np
    states_array = np.array(states)
    encoder.fit(states_array)
    
    # Save encoder
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    encoder.save(save_path)
    
    # Print cluster info
    info = encoder.get_cluster_info()
    print(f"   - {info['n_clusters']} clusters discovered")
    print(f"   - {info['total_variance_explained']*100:.1f}% variance explained")
    
    return encoder


def main():
    parser = argparse.ArgumentParser(description='Train Ghost RL Agent')
    parser.add_argument('--config', type=str, default='training/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save-path', type=str, default='models/ghost_agent',
                       help='Path to save trained model')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Override total training timesteps')
    parser.add_argument('--no-encoder', action='store_true',
                       help='Skip state encoder training')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    if os.path.exists(args.config):
        config_path = args.config
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Override timesteps if provided
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps
    
    # Train ghost agent
    agent = train_ghost_agent(config, args.save_path)
    
    # Optionally train state encoder
    if not args.no_encoder:
        env = create_training_env(agent_type='ghost', n_envs=1)
        states = collect_state_data(env, num_episodes=50)
        encoder = train_state_encoder(
            states, config,
            save_path=os.path.join(args.save_path, 'state_encoder.pkl')
        )
    
    print("\n✓ All training complete!")


if __name__ == '__main__':
    main()
