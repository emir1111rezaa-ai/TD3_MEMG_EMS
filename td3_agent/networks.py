import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor Network for TD3 Agent
    Maps state to continuous action
    Architecture: State -> FC -> ReLU -> FC -> ReLU -> FC -> Tanh
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights
        self.fc1.weight.data.uniform_(-1/torch.sqrt(torch.tensor(state_dim)), 
                                       1/torch.sqrt(torch.tensor(state_dim)))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.max_action * x


class Critic(nn.Module):
    """
    Critic Network for TD3 Agent (Dueling Architecture - Two Q-networks)
    Maps (state, action) to Q-value
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # First Q-network
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Second Q-network
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        
        # First Q-network
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Second Q-network
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Return only Q1 for computational efficiency"""
        sa = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1
