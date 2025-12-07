import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) Agent
    
    Key Features:
    1. Twin Q-networks (Double Q-learning) to reduce overestimation bias
    2. Delayed policy update (update actor less frequently than critic)
    3. Target policy smoothing (add noise to target actions)
    4. Separate target networks for stability
    
    Reference: Fujimoto et al., "Addressing Function Approximation Error in 
    Actor-Critic Methods" (ICML 2018)
    """
    
    def __init__(self, state_dim, action_dim, max_action=1.0, 
                 hidden_dim=256, lr_actor=3e-4, lr_critic=3e-4,
                 gamma=0.99, tau=0.005, device='cpu'):
        """
        Initialize TD3 Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action magnitude (for scaling)
            hidden_dim: Hidden layer size for networks
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.total_it = 0
        self.delay_period = 2  # Update actor every 2 critic updates
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks (dual Q-networks)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device=device)
        
        # Policy noise for exploration and target smoothing
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
    
    def select_action(self, state, eval_mode=False):
        """
        Select action from policy
        
        Args:
            state: Current state (numpy array)
            eval_mode: If True, no exploration noise
        
        Returns:
            Action (numpy array)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if not eval_mode:
            # Add exploration noise during training
            noise = np.random.normal(0, self.policy_noise, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def train(self, batch_size=256):
        """
        Train TD3 agent using one batch from replay buffer
        
        Args:
            batch_size: Size of training batch
        
        Returns:
            Dictionary with training metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        self.total_it += 1
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # ============ CRITIC UPDATE ============
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.normal(0, self.policy_noise, size=actions.size()).to(self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            
            # Compute target actions
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.max_action, self.max_action)
            
            # Compute target Q-values (take minimum of two Q-networks)
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Compute target Q-value
            target_Q = rewards + (1 - dones) * self.gamma * target_Q
        
        # Get current Q-values
        current_Q1, current_Q2 = self.critic(states, actions)
        
        # Critic loss (MSE between predicted and target Q-values)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ============ DELAYED ACTOR UPDATE ============
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': 0.0,
            'q_mean': target_Q.mean().item()
        }
        
        if self.total_it % self.delay_period == 0:
            # Compute actor loss (negative Q-value)
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ============ SOFT UPDATE TARGET NETWORKS ============
            # Update target networks using exponential moving average
            for param, target_param in zip(self.critic.parameters(), 
                                          self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), 
                                          self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            metrics['actor_loss'] = actor_loss.item()
        
        return metrics
    
    def save_checkpoint(self, filepath):
        """
        Save agent checkpoint
        """
        checkpoint = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath):
        """
        Load agent checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']
        print(f"Checkpoint loaded from {filepath}")
