import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from td3_agent.td3_core import TD3Agent
from memg_env.microgrid import MEMGEnvironment
from reward_function.cvar_reward import RiskAwareEMSReward


class TD3Trainer:
    """
    Training loop for TD3 Agent on MEMG Environment
    """
    
    def __init__(self, state_dim=10, action_dim=3, hidden_dim=256,
                 batch_size=256, device='cpu', save_dir='./checkpoints'):
        """
        Initialize trainer
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer size
            batch_size: Training batch size
            device: 'cpu' or 'cuda'
            save_dir: Directory to save checkpoints
        """
        self.device = device
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize agent and environment
        self.agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        
        self.env = MEMGEnvironment(forecast_horizon=24)  # 24 timesteps = 6 hours (15 min each)
        self.reward_function = RiskAwareEMSReward(alpha_cvar=0.95)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_violations = []
        self.actor_losses = []
        self.critic_losses = []
        self.q_values = []
    
    def generate_synthetic_profiles(self, episode_length=24):
        """
        Generate synthetic load and renewable profiles for an episode
        Simple sinusoidal + random variations
        
        Args:
            episode_length: Number of timesteps
        
        Returns:
            pv_profile, wt_profile, load_profile, heat_load_profile
        """
        t = np.arange(episode_length)
        
        # PV profile (peaks at noon)
        pv_base = 30.0 * np.sin(np.pi * t / (episode_length - 1))
        pv_profile = np.maximum(pv_base + np.random.normal(0, 3, episode_length), 0)
        pv_profile = np.clip(pv_profile, 0, 60)  # Max 60 kW
        
        # Wind profile (more variable)
        wt_profile = 20 + 15 * np.sin(2 * np.pi * t / episode_length) + np.random.normal(0, 5, episode_length)
        wt_profile = np.clip(wt_profile, 0, 40)  # Max 40 kW
        
        # Electric load (peaks in morning/evening)
        load_peaks = np.array([0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.5, 0.4, 0.3,
                              0.2, 0.1, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5])
        if episode_length < 24:
            load_peaks = load_peaks[:episode_length]
        load_profile = 30 + load_peaks[:episode_length] * 50 + np.random.normal(0, 3, episode_length)
        load_profile = np.clip(load_profile, 10, 100)
        
        # Thermal load
        heat_load = 40 + 30 * np.sin(2 * np.pi * t / episode_length) + np.random.normal(0, 5, episode_length)
        heat_load = np.clip(heat_load, 15, 150)
        
        return pv_profile, wt_profile, load_profile, heat_load
    
    def train_episode(self):
        """
        Train for one full episode
        
        Returns:
            Episode statistics
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_violations = 0
        
        # Generate profiles for this episode
        pv_prof, wt_prof, load_prof, heat_prof = self.generate_synthetic_profiles(24)
        
        for step in range(24):
            # Select action
            action = self.agent.select_action(state, eval_mode=False)
            
            # Step environment
            next_state, reward, done, info = self.env.step(
                action,
                pv_prof[step],
                wt_prof[step],
                load_prof[step],
                heat_prof[step]
            )
            
            # Store experience in replay buffer
            self.agent.replay_buffer.add(state, action, reward, next_state, float(done))
            
            # Update statistics
            episode_reward += reward
            episode_cost += info['cost']
            episode_violations += info['constraint_violations']
            
            # Train agent
            train_metrics = self.agent.train(batch_size=self.batch_size)
            
            if train_metrics:
                self.critic_losses.append(train_metrics.get('critic_loss', 0))
                self.actor_losses.append(train_metrics.get('actor_loss', 0))
                self.q_values.append(train_metrics.get('q_mean', 0))
            
            state = next_state
            
            if done:
                break
        
        # Record episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_costs.append(episode_cost)
        self.episode_violations.append(episode_violations)
        
        return {
            'episode_reward': episode_reward,
            'episode_cost': episode_cost,
            'constraint_violations': episode_violations,
            'avg_cost_per_step': episode_cost / 24,
            'violation_rate': episode_violations / 24
        }
    
    def train(self, num_episodes=50, eval_interval=10, checkpoint_interval=10):
        """
        Main training loop
        
        Args:
            num_episodes: Number of training episodes
            eval_interval: Evaluate every N episodes
            checkpoint_interval: Save checkpoint every N episodes
        """
        print(f"\n{'='*70}")
        print(f"Starting TD3 Training on MEMG Environment")
        print(f"Episodes: {num_episodes} | Device: {self.device}")
        print(f"{'='*70}\n")
        
        for episode in range(num_episodes):
            stats = self.train_episode()
            
            # Log progress
            if (episode + 1) % 5 == 0:
                avg_reward_recent = np.mean(self.episode_rewards[-5:])
                avg_cost_recent = np.mean(self.episode_costs[-5:])
                avg_violations_recent = np.mean(self.episode_violations[-5:])
                
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Reward: {avg_reward_recent:.2f} | "
                      f"Cost: {avg_cost_recent:.2f}€ | "
                      f"Violations: {avg_violations_recent:.1f}")
            
            # Save checkpoint
            if (episode + 1) % checkpoint_interval == 0:
                self.save_checkpoint(episode + 1)
                print(f"  ✓ Checkpoint saved at episode {episode + 1}")
        
        print(f"\n{'='*70}")
        print(f"Training completed!")
        print(f"Final metrics:")
        print(f"  - Mean Reward: {np.mean(self.episode_rewards[-10:]):.2f}")
        print(f"  - Mean Cost: {np.mean(self.episode_costs[-10:]):.2f}€")
        print(f"  - Mean Violations: {np.mean(self.episode_violations[-10:]):.1f}")
        print(f"{'='*70}\n")
    
    def evaluate(self, num_episodes=5):
        """
        Evaluate trained agent (no exploration noise)
        """
        eval_rewards = []
        eval_costs = []
        eval_violations = []
        
        print(f"\nEvaluating agent over {num_episodes} episodes...\n")
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_cost = 0.0
            episode_violations = 0
            
            pv_prof, wt_prof, load_prof, heat_prof = self.generate_synthetic_profiles(24)
            
            for step in range(24):
                action = self.agent.select_action(state, eval_mode=True)  # No noise
                
                next_state, reward, done, info = self.env.step(
                    action, pv_prof[step], wt_prof[step], load_prof[step], heat_prof[step]
                )
                
                episode_reward += reward
                episode_cost += info['cost']
                episode_violations += info['constraint_violations']
                
                state = next_state
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_costs.append(episode_cost)
            eval_violations.append(episode_violations)
            
            print(f"Eval Episode {episode+1}: Cost={episode_cost:.2f}€, "
                  f"Violations={episode_violations}, Reward={episode_reward:.2f}")
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'mean_cost': np.mean(eval_costs),
            'mean_violations': np.mean(eval_violations),
            'std_cost': np.std(eval_costs),
            'max_cost': np.max(eval_costs)
        }
    
    def save_checkpoint(self, episode):
        """
        Save training checkpoint
        """
        checkpoint_path = self.save_dir / f"td3_episode_{episode}.pt"
        self.agent.save_checkpoint(str(checkpoint_path))
        
        # Save statistics
        stats = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs,
            'episode_violations': self.episode_violations,
            'timestamp': datetime.now().isoformat()
        }
        
        stats_path = self.save_dir / f"stats_episode_{episode}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint
        """
        self.agent.load_checkpoint(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
